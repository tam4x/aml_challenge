import argparse
from PIL import Image
import torch
#from sentence_transformers import SentenceTransformer
#from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
import random
# roberta-large-nli-stsb-mean-tokens
def load_text_model(model_name="sentence-transformers/roberta-large-nli-stsb-mean-tokens"):
    """Load Sentence-BERT text encoder."""
    print(f"Loading text model: {model_name}")
    return SentenceTransformer(model_name)


def load_image_model(model_name="facebook/dinov2-giant"):
    """Load DINOv2 image encoder."""
    print(f"Loading image model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    return image_processor, model

## Can be modified for data augmentation if needed or create my one
def get_augmentations(num_augmentations=3):
    """Create a list of augmentation pipelines (light + heavy mix)."""
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

    augs = []
    for _ in range(num_augmentations):
        augs.append(random.choice([light, heavy]))
    return augs

@torch.inference_mode()
def process_images_batch(image_processor, model, image_paths, device, batch_size=128, dataset_path=None, augmentations=3):
    """Generate image embeddings in batches."""
    print(f"Processing {len(image_paths)} images in batches...")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    all_names = []
    img_files = []
    augs = get_augmentations(augmentations)

    for i, path in enumerate(tqdm(image_paths, desc="Encoding augmented images")):
        try:
            img = Image.open(dataset_path / 'Images' / path).convert("RGB")
        except Exception as e:
            print(f"Warning: Skipping image {path} due to error: {e}")
            continue

        for k, aug in enumerate(augs):
            aug_img = aug(img)
            inputs = image_processor(images=[aug_img], return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(emb)
            all_names.append(f"{path}_aug{k+1}")

    all_embeddings = np.vstack(all_embeddings)

    return all_names, all_embeddings

    
def process_captions(text_model, captions, device):
    """Generate text embeddings using Sentence-BERT."""
    print("Processing captions...")
    return text_model.encode(
        captions, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        device=device
    )

def load_dataset(dataset_path):
    """
    Load dataset from a directory containing captions.txt and an Images folder.
    """
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "Images"
    if not captions_file.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Could not find 'captions.txt' or 'Images' directory in {dataset_path}")

    df = pd.read_csv(captions_file)
    
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))

    return df

def create_data_file(dataset_path, output_file, device=None, args={}):
    """
    Main function to generate embeddings and save the final .npz file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset from: {dataset_path}")
    df_captions = load_dataset(dataset_path)

    text_model = load_text_model()
    image_processor, image_model = load_image_model()

    num_augmentations = getattr(args, "num_augmentations", 3)

    all_captions = df_captions['caption'].tolist()
    caption2img = df_captions['image'].tolist()
    all_images = df_captions['image'].unique().tolist()

    num_images = len(all_images)
    num_captions = len(all_captions)
    print(f"Found {num_images} images and {num_captions} total captions.")

    all_images, img_embd = process_images_batch(image_processor, 
                                                image_model, 
                                                all_images, 
                                                device, 
                                                dataset_path=dataset_path, augmentations=num_augmentations)
    
    images_dict = {img_name: i for i, img_name in enumerate(all_images)}

    # Duplicate captions per image *and* per augmentation
    repeated_captions = []
    repeated_imgnames = []
    for img_name, caption in zip(df_captions["image"], df_captions["caption"]):
        for k in range(num_augmentations):
            repeated_captions.append(caption)
            repeated_imgnames.append(f"{img_name}_aug{k+1}")

    caption_embeddings = process_captions(text_model, repeated_captions, device)

    num_images = len(all_images)
    num_captions = len(repeated_captions)

    label = np.zeros((num_captions, num_images), dtype=bool)
    image_idx_map = {name: i for i, name in enumerate(all_images)}
    for idx, img_name in enumerate(repeated_imgnames):
        if img_name in image_idx_map:
            label[idx, image_idx_map[img_name]] = 1

    data = {
        'metadata/num_captions': np.array([num_captions]),
        'metadata/num_images': np.array([num_images]),
        'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
        'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
        'captions/text': np.array(repeated_captions),
        'captions/embeddings': caption_embeddings,
        'captions/label': label,
        'images/names': np.array(all_images),
        'images/embeddings': img_embd,
    }

    print(f"Saving augmented dataset to {output_file}")
    np.savez_compressed(output_file, **data)
    print("✓ Done.")
    
    
def main():
    parser = argparse.ArgumentParser(description="Preprocess image-caption dataset with augmentations and save to .npz.")
    parser.add_argument("input_folder", type=Path, help="Path to dataset folder (e.g. data/train)")
    parser.add_argument("--output-file", "-o", type=str, default="processed_augmented_data.npz", help="Output .npz path")
    parser.add_argument("--num-augmentations", type=int, default=3, help="Number of augmentations per image (e.g. 3 or 5)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    create_data_file(args.input_folder, args.output_file, args.device, args)

if __name__ == "__main__":
    main()
