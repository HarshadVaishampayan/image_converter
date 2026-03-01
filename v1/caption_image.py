from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    Blip2ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Dict, Any, Generator
from persistence import PersistenceAPI
from model_manager import ModelManager, detect_environment, load_models
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import time
import traceback
import argparse


# ------------------------------
# Helper Functions and Classes
# ------------------------------
def _get_image_paths(path: str) -> List[str]:
    """Helper function to recursively get all image paths."""
    image_paths = []
    if os.path.isfile(path):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, file))
    return image_paths


class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading images from a list of paths."""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            return image, image_path
        except Exception as e:
            # Return None for path to indicate failure
            return None, image_path


def collate_fn(batch):
    """Custom collate function to filter out items where the image failed to load."""
    # A batch is a list of tuples (image, path)
    # Filter out tuples where the image is None
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    # Unzip the filtered batch
    images, paths = zip(*batch)
    return list(images), list(paths)


# ------------------------------
# Main Processing Function
# ------------------------------
def process_folder_parallel_blip(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = None,
) -> Generator:
    """
    Process images in a directory recursively using BLIP for captioning,
    with a parallel data loading pipeline.
    """
    total_start_time = time.time()
    print(f"\nProcessing path: {path}")

    # Get cached models and parameters
    model_manager = ModelManager()
    processor, model, beam_size, batch_size = model_manager.get_blip_models()
    clip_model = model_manager.get_clip_model()
    device = model_manager.device

    # Get all image paths recursively
    paths_to_process = _get_image_paths(path)
    if not paths_to_process:
        print(f"Error: No valid images found in {path}")
        return

    # Setup DataLoader for parallel loading
    # Use a small, fixed number of workers to prevent memory thrashing from aggressive prefetching
    num_workers = 1 
    
    dataset = ImageDataset(paths_to_process)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if "cuda" in str(device) else False,
        prefetch_factor=2,
    )

    print(f"Found {len(paths_to_process)} images. Starting processing with batch size {batch_size} and {num_workers} data loader workers.")
    
    processed_count = 0
    loop_start_time = time.time()

    for batch_images, batch_paths in data_loader:
        batch_load_time = time.time()
        
        if not batch_images:
            print("[WARNING] DataLoader returned an empty batch. Skipping.")
            continue

        print(f"\n--- Loaded batch of {len(batch_images)} images in {batch_load_time - loop_start_time:.2f}s ---")

        try:
            # --- Preprocessing and Model Inference with Mixed Precision ---
            preprocessing_start_time = time.time()
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            preprocessing_end_time = time.time()
            print(f"  - Preprocessing for batch took {preprocessing_end_time - preprocessing_start_time:.2f}s")

            inference_start_time = time.time()
            with torch.autocast(device_type=str(device).split(":")[0]):
                # Generate captions using BLIP
                outputs = model.generate(
                    **inputs, num_beams=beam_size, num_return_sequences=beam_size, max_new_tokens=100
                )
                
                # Generate embeddings using CLIP
                embeddings = clip_model.encode(
                    batch_images, convert_to_tensor=True, device=device
                )

            # Decode captions for each image in the batch
            all_captions = [
                processor.decode(output, skip_special_tokens=True) for output in outputs
            ]
            embeddings = embeddings.cpu().numpy()

            inference_end_time = time.time()
            print(f"  - Inference for batch took {inference_end_time - inference_start_time:.2f}s")

            # --- Process Results ---
            # Group captions by image (since num_return_sequences > 1)
            captions_per_image = [
                all_captions[j : j + beam_size]
                for j in range(0, len(all_captions), beam_size)
            ]

            for idx, image_path in enumerate(batch_paths):
                image = batch_images[idx]
                captions = captions_per_image[idx]
                embedding = embeddings[idx]

                result = {
                    "image_path": image_path,
                    "captions": [{"text": cap} for cap in captions],
                    "embedding": embedding,
                    "width": image.width,
                    "height": image.height,
                    "metadata": {
                        "env": detect_environment(),
                        "device": str(device),
                        "batch_size": batch_size,
                        "beam_size": beam_size,
                    },
                }

                processed_count += 1

                if consumers:
                    for consumer in consumers:
                        try:
                            consumer(result)
                        except Exception as e:
                            print(
                                f"[WARNING] Consumer {consumer.__name__} failed for {image_path}: {e}"
                            )
                            traceback.print_exc()

                if cache_file:
                    with open(cache_file, "a") as f:
                        json.dump(
                            {"image_path": result["image_path"], "captions": result["captions"]}, f
                        )
                        f.write("\n")

                yield result

        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            traceback.print_exc()
        
        loop_start_time = time.time() # Reset timer for the next batch load measurement

    total_end_time = time.time()
    print(f"\n--- Processing Summary ---")
    print(f"Processed {processed_count} images in {total_end_time - total_start_time:.2f} seconds.")


def normalize(embedding):
    """
    Normalize embedding vector to unit length.
    Handles both numpy arrays and pytorch tensors.
    """
    import torch
    import numpy as np
    
    if isinstance(embedding, np.ndarray):
        # Numpy array normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    elif isinstance(embedding, torch.Tensor):
        # PyTorch tensor normalization
        return torch.nn.functional.normalize(embedding, p=2, dim=-1)
    else:
        raise TypeError(f"Unsupported embedding type: {type(embedding)}")

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_embeddings(clip_model, device, images):
    batch_embeddings = clip_model.encode(images, convert_to_tensor=True, device=device)
    return batch_embeddings

def blip_caption_single(img: Image.Image, blip_processor: BlipProcessor, blip_model: BlipForConditionalGeneration, beam_size: int, device: torch.device):
    """Generate captions for a single image using BLIP."""
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, num_beams=beam_size, num_return_sequences=beam_size, max_new_tokens=100)
    captions = [blip_processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# ------------------------------
# Example consumer
# ------------------------------
def print_consumer(data: Dict[str, Any]):
    """Simple consumer that prints output."""
    print("PRINT:", data["image_path"], "→", [c["text"] for c in data.get("captions", [])])

# ------------------------------
# Persistence consumer
# ------------------------------
def make_persistence_consumer(db_path: str = "images.db", dim: int = 512):
    """
    Returns a consumer that writes image, captions, metadata, and embedding
    into SQLite (metadata) and FAISS (vector search) via PersistenceAPI.
    """
    api = PersistenceAPI(db_path=db_path, dim=dim)

    def _consumer(data: Dict[str, Any]):
        file_path = data["image_path"]
        emb = data["embedding"]
        captions = [c["text"] for c in data.get("captions", [])]
        width = data.get("width")
        height = data.get("height")
        metadata = data.get("metadata", {})

        # Convert numpy embedding → Python list[float] for PersistenceAPI
        emb_list = emb.tolist() if hasattr(emb, "tolist") else emb

        api.insert_or_update(
            file_path=file_path,
            embedding=emb_list,
            captions=captions,
            metadata=metadata,
            width=width,
            height=height,
        )

    return _consumer

# ------------------------------
# Usage
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Describe an image using Ollama or Hugging Face models.')
    parser.add_argument('--ip', required=True, dest='image_path', help='Path to the image file')

    args = parser.parse_args()

    print("-START-")
    output_lines = []
    start_str = f"START: {args.image_path}"

    cache = "processed_cache.jsonl"
    # Add the persistence consumer alongside the print consumer
    persist_consumer = make_persistence_consumer(db_path="images.db", dim=512)
    for _ in process_folder_parallel_blip(
        path=args.image_path,
        consumers=[print_consumer, persist_consumer],
        cache_file=cache,
    ):
        pass


def generate_caption_blip2_coco(image_path: str, max_new_tokens: int = 20) -> str:
    model_manager = ModelManager()
    processor, model = model_manager.get_blip2_models()
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model_manager.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated, skip_special_tokens=True)[0]
    
    return caption

