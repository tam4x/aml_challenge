import argparse
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
import random
import os
import json

# --- MODELLI (invariato) ---
def load_text_model(model_name="sentence-transformers/roberta-large-nli-stsb-mean-tokens"):
    """Carica il codificatore di testo Sentence-BERT."""
    print(f"Caricamento modello di testo: {model_name}")
    return SentenceTransformer(model_name)

def load_image_model(model_name="facebook/dinov2-giant"):
    """Carica il codificatore di immagini DINOv2."""
    print(f"Caricamento modello di immagini: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    return image_processor, model

# --- AUGMENTATIONS (invariato) ---
def get_augmentations(num_augmentations=3):
    """Crea una lista di pipeline di augmentation (mix leggero + pesante)."""
    light = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
    ])
    heavy = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(),
    ])
    augs = [random.choice([light, heavy]) for _ in range(num_augmentations)]
    return augs

# --- OTTIMIZZAZIONE 1: ELABORAZIONE IMMAGINI A LOTTI (BATCHING) E CHECKPOINTING ---
@torch.inference_mode()
def process_images_with_checkpointing(
    image_processor, model, image_paths, device, dataset_path,
    checkpoint_dir, num_augmentations=3, batch_size=64
):
    """
    Genera embedding di immagini in batch, con logica di checkpoint per la resilienza.
    Salva i progressi dopo ogni batch e carica i dati se già presenti.
    """
    print("Avvio elaborazione immagini...")
    model.to(device)
    model.eval()

    # Definisci i percorsi per i file di checkpoint
    names_chk_path = checkpoint_dir / "image_names.json"
    embeds_chk_path = checkpoint_dir / "image_embeddings.npy"

    # Se i checkpoint finali esistono, carica e restituisci i dati
    if names_chk_path.exists() and embeds_chk_path.exists():
        print(f"✓ Trovati checkpoint completi. Caricamento embedding immagini da '{checkpoint_dir}'...")
        with open(names_chk_path, 'r') as f:
            all_names = json.load(f)
        all_embeddings = np.load(embeds_chk_path)
        return all_names, all_embeddings

    # Logica di ripresa dal crash
    processed_images = set()
    all_names = []
    all_embeddings_list = []

    # Controlla se esistono checkpoint parziali (modalità 'append')
    if names_chk_path.exists():
        with open(names_chk_path, 'r') as f:
            all_names = json.load(f)
        # Estrai i nomi delle immagini originali già processate
        # E.g., da "image_01.jpg_aug1" estrai "image_01.jpg"
        processed_images = set(name.rsplit('_aug', 1)[0] for name in all_names)
        print(f"Ripresa dal checkpoint: {len(processed_images)} immagini uniche già elaborate.")
    
    if embeds_chk_path.exists():
        all_embeddings_list = [np.load(embeds_chk_path)]

    # Filtra le immagini che devono ancora essere processate
    images_to_process = [p for p in image_paths if p not in processed_images]
    if not images_to_process:
        print("✓ Tutte le immagini sono già state elaborate.")
        return all_names, np.vstack(all_embeddings_list) if all_embeddings_list else np.array([])

    print(f"Elaborazione di {len(images_to_process)} immagini rimanenti in lotti da {batch_size}...")
    
    augs = get_augmentations(num_augmentations)
    
    # Itera sulle immagini da processare in batch
    for i in tqdm(range(0, len(images_to_process), batch_size), desc="Encoding immagini"):
        batch_paths = images_to_process[i:i+batch_size]
        batch_augmented_images = []
        batch_names = []

        # Prepara il batch: carica, applica augmentations
        for path in batch_paths:
            try:
                img = Image.open(dataset_path / 'Images' / path).convert("RGB")
                for k, aug in enumerate(augs):
                    aug_img = aug(img)
                    batch_augmented_images.append(aug_img)
                    batch_names.append(f"{path}_aug{k+1}")
            except Exception as e:
                print(f"Attenzione: Saltata immagine {path} a causa di un errore: {e}")
                continue
        
        if not batch_augmented_images:
            continue

        # Processa l'intero batch di immagini aumentate in un colpo solo
        inputs = image_processor(images=batch_augmented_images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Pooler output o media dell'ultimo stato nascosto
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Aggiungi i risultati alle liste principali
        all_names.extend(batch_names)
        all_embeddings_list.append(emb)

        # --- SALVATAGGIO CHECKPOINT DOPO OGNI BATCH ---
        # Sovrascrive i file con i dati aggiornati. È più semplice che usare la modalità append.
        temp_embeds = np.vstack(all_embeddings_list)
        np.save(embeds_chk_path, temp_embeds)
        with open(names_chk_path, 'w') as f:
            json.dump(all_names, f)

    print("✓ Elaborazione immagini completata.")
    final_embeddings = np.vstack(all_embeddings_list)
    return all_names, final_embeddings

# --- OTTIMIZZAZIONE 2: CHECKPOINTING PER LE DIDASCALIE ---
def process_captions_with_checkpointing(text_model, captions, device, checkpoint_dir):
    """
    Genera embedding di testo, con logica di checkpoint.
    """
    print("Avvio elaborazione didascalie...")
    
    # Definisci i percorsi per i file di checkpoint
    embeds_chk_path = checkpoint_dir / "caption_embeddings.npy"

    if embeds_chk_path.exists():
        print(f"✓ Trovato checkpoint. Caricamento embedding didascalie da '{embeds_chk_path}'...")
        return np.load(embeds_chk_path)

    print("Nessun checkpoint trovato, calcolo degli embedding delle didascalie...")
    embeddings = text_model.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
        batch_size=128 # Aggiunto batch_size per efficienza
    )
    
    # Salva il checkpoint una volta completato
    print(f"Salvataggio checkpoint embedding didascalie su '{embeds_chk_path}'...")
    np.save(embeds_chk_path, embeddings)
    
    print("✓ Elaborazione didascalie completata.")
    return embeddings

# --- FUNZIONI DI SUPPORTO (invariato) ---
def load_dataset(dataset_path):
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "Images"
    if not captions_file.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Impossibile trovare 'captions.txt' o la cartella 'Images' in {dataset_path}")
    df = pd.read_csv(captions_file)
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))
    return df

# --- FUNZIONE PRINCIPALE AGGIORNATA ---
def create_data_file(args):
    """
    Funzione principale per generare embedding e salvare il file .npz finale.
    """
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # Crea la directory per i checkpoint se non esiste
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"I checkpoint verranno salvati in: '{checkpoint_dir}'")

    print(f"Caricamento dataset da: {args.input_folder}")
    df_captions = load_dataset(args.input_folder)

    # Carica i modelli
    text_model = load_text_model()
    image_processor, image_model = load_image_model()

    # Estrai lista di immagini uniche
    unique_images = df_captions['image'].unique().tolist()
    print(f"Trovate {len(unique_images)} immagini uniche e {len(df_captions)} didascalie totali.")

    # ... (dentro create_data_file)
    # 1. Processa le immagini con la nuova funzione ottimizzata e con checkpoint
    # Questa funzione restituisce SOLO le immagini elaborate con successo.
    augmented_image_names, img_embd = process_images_with_checkpointing(
        image_processor, image_model, unique_images, device,
        dataset_path=args.input_folder,
        checkpoint_dir=checkpoint_dir,
        num_augmentations=args.num_augmentations,
        batch_size=args.batch_size
    )

    # 2. CREA UN SET DI IMMAGINI ORIGINALI PROCESSATE CON SUCCESSO
    # Questa è la nostra "fonte di verità" per sapere quali immagini sono valide.
    successful_original_images = set(name.rsplit('_aug', 1)[0] for name in augmented_image_names)
    print(f"Trovate {len(successful_original_images)} immagini originali elaborate con successo su {len(unique_images)} totali.")

    # 3. RICOSTRUISCI LA LISTA DELLE DIDASCALIE BASANDOTI SOLO SULLE IMMAGINI VALIDE
    print("Accoppiamento Sincronizzato: didascalie -> immagini aumentate valide...")
    repeated_captions = []
    repeated_imgnames = []
    # Itera sul dataframe originale per mantenere l'ordine e l'associazione
    for _, row in tqdm(df_captions.iterrows(), total=len(df_captions), desc="Pairing captions"):
        img_name, caption = row['image'], row['caption']
        # Includi questa didascalia solo se la sua immagine originale è stata processata
        if img_name in successful_original_images:
            for k in range(args.num_augmentations):
                repeated_captions.append(caption)
                # Associa ogni didascalia ripetuta al nome dell'immagine aumentata corrispondente
                repeated_imgnames.append(f"{img_name}_aug{k+1}")

    # 4. Processa le didascalie (ora la lista 'repeated_captions' è corretta)
    caption_embeddings = process_captions_with_checkpointing(
        text_model, repeated_captions, device, checkpoint_dir
    )

    # 5. Crea la matrice delle etichette (la logica qui ora funzionerà correttamente)
    print("Creazione della matrice delle etichette...")
    num_augmented_images = len(augmented_image_names)
    num_repeated_captions = len(repeated_captions)
    
    label = np.zeros((num_repeated_captions, num_augmented_images), dtype=bool)
    image_idx_map = {name: i for i, name in enumerate(augmented_image_names)}
    
    # Usa la lista 'repeated_imgnames' che è già sincronizzata
    for idx, img_name in enumerate(tqdm(repeated_imgnames, desc="Building labels")):
        if img_name in image_idx_map:
            label[idx, image_idx_map[img_name]] = 1
    
    # ... resto dello script (assemblaggio e salvataggio)

    # 4. Crea la matrice delle etichette (veloce, non necessita di checkpoint)
    print("Creazione della matrice delle etichette...")
    num_augmented_images = len(augmented_image_names)
    num_repeated_captions = len(repeated_captions)
    
    label = np.zeros((num_repeated_captions, num_augmented_images), dtype=bool)
    image_idx_map = {name: i for i, name in enumerate(augmented_image_names)}
    
    for idx, img_name in enumerate(tqdm(repeated_imgnames, desc="Building labels")):
        if img_name in image_idx_map:
            label[idx, image_idx_map[img_name]] = 1

    # 5. Assembla e salva il file finale
    data = {
        'metadata/num_captions': np.array([num_repeated_captions]),
        'metadata/num_images': np.array([num_augmented_images]),
        'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
        'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
        'captions/text': np.array(repeated_captions),
        'captions/embeddings': caption_embeddings,
        'captions/label': label,
        'images/names': np.array(augmented_image_names),
        'images/embeddings': img_embd,
    }

    print(f"Salvataggio del dataset aumentato su {args.output_file}")
    np.savez_compressed(args.output_file, **data)
    print("✓ Fatto.")
    print(f"Puoi eliminare la cartella dei checkpoint '{checkpoint_dir}' se non ti serve più.")

def main():
    parser = argparse.ArgumentParser(description="Preprocessa un dataset immagine-didascalia con augmentation e lo salva in .npz.")
    parser.add_argument("input_folder", type=Path, help="Percorso alla cartella del dataset (es. data/train)")
    parser.add_argument("--output-file", "-o", type=str, default="processed_augmented_data.npz", help="Percorso del file .npz di output")
    parser.add_argument("--num-augmentations", type=int, default=3, help="Numero di augmentation per immagine")
    parser.add_argument("--batch-size", type=int, default=64, help="Dimensione del batch per l'elaborazione delle immagini")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Cartella per salvare e caricare i checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo da usare (es. 'cuda', 'cpu')")
    args = parser.parse_args()

    create_data_file(args)

if __name__ == "__main__":
    main()