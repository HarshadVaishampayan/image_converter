from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Dict, Any, Generator
from persistence import PersistenceAPI
from model_manager import ModelManager, detect_environment
from describe_image import describe_image # Import the existing Ollama integration
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import time
import traceback
import argparse
import re
from metrics import Metrics

# ------------------------------
# Helper Functions and Classes
# ------------------------------
def cleanup_text(text: str) -> str:
    """Removes common preambles, special tokens, and markdown formatting from model outputs."""
    if not text:
        return ""
    
    # 1. Remove special tokens commonly found in model outputs
    text = re.sub(r'<pad>|<s>|</s>|<\|endoftext\|>', '', text)
    
    # 2. Remove any remaining <TAGS> (common in Florence-2 raw output)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Remove Markdown formatting (bold, italics, code blocks)
    text = re.sub(r'\*\*|\*|__|`', '', text)
    
    # 4. Remove common preambles (case-insensitive)
    preambles = [
        r"^the image shows\s*(?:a|an|the)?\s*",
        r"^the image is\s*(?:a|an|the)?\s*",
        r"^this image shows\s*(?:a|an|the)?\s*",
        r"^this is an image of\s*(?:a|an|the)?\s*",
        r"^here is a factual description of the image:?\s*",
        r"^factual description:?\s*",
        r"^description:?\s*",
        r"^image description:?\s*",
    ]
    
    for pattern in preambles:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 5. Capitalize first letter and strip whitespace
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        
    return text

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
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return list(images), list(paths)


# ------------------------------
# Main Processing Function
# ------------------------------
def process_folder_parallel_caption(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = None,
    embedding_model_type: str = "siglip",
    caption_model_type: str = "ollama"
) -> Generator:
    """
    Process images in a directory recursively using:
    1. Selected caption model (Florence-2 or Ollama) for description.
    2. Selected embedding model (CLIP/SigLIP) for vectors.
    """
    metrics = Metrics()
    metrics.reset()
    
    total_start_time = time.time()
    print(f"\nProcessing path: {path} | Embed: {embedding_model_type.upper()} | Cap: {caption_model_type.upper()}")

    # Get cached models and parameters
    model_manager = ModelManager()
    
    # Lazy-load Florence-2 only if requested
    caption_processor, caption_model, beam_size, batch_size = (None, None, 1, 2)
    if caption_model_type == "florence2":
        caption_processor, caption_model, beam_size, batch_size = model_manager.get_caption_model()
    
    # Selected embedding model
    embedding_model = model_manager.get_embedding_model(embedding_model_type)
    device = model_manager.device

    # Get all image paths recursively
    paths_to_process = _get_image_paths(path)
    if not paths_to_process:
        print(f"Error: No valid images found in {path}")
        return

    # Setup DataLoader
    num_workers = 0 
    dataset = ImageDataset(paths_to_process)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False, 
    )

    print(f"Found {len(paths_to_process)} images. Starting processing with batch size {batch_size}.")
    
    processed_count = 0
    total_images = len(paths_to_process)
    
    # Task Prompts
    florence_prompt = "<MORE_DETAILED_CAPTION>"

    metrics.start("Image Fetching (Disk)")
    for batch_images, batch_paths in data_loader:
        metrics.stop("Image Fetching (Disk)")
        batch_load_time = time.time()
        
        if not batch_images:
            metrics.start("Image Fetching (Disk)")
            print("[WARNING] DataLoader returned an empty batch. Skipping.")
            continue

        batch_start_idx = processed_count + 1
        batch_end_idx = min(processed_count + len(batch_images), total_images)
        print(f"\n>>> Processing images {batch_start_idx}-{batch_end_idx} of {total_images} <<<")
        print(f"--- Loaded batch of {len(batch_images)} images ---")
        metrics.start("Batch Total Processing")

        try:
            # --- 1. Embedding Inference (Universal) ---
            with torch.no_grad():
                print(f"  Generating {embedding_model_type.upper()} embeddings...")
                metrics.start("Image Embedding")
                embeddings = embedding_model.encode(
                    batch_images, convert_to_tensor=True, device=device
                )
                metrics.stop("Image Embedding")

            embeddings_np = embeddings.cpu().numpy()
            del embeddings

            # --- 2. Image Description (Model Specific) ---
            final_descriptions = ["" for _ in range(len(batch_paths))]
            
            if caption_model_type == "florence2":
                print("  Generating descriptions via Florence-2...")
                metrics.start("Florence-2 Inference")
                
                florence_prompts = [florence_prompt] * len(batch_images)
                florence_inputs = caption_processor(text=florence_prompts, images=batch_images, return_tensors="pt")
                
                # Ensure inputs match model's precision
                model_dtype = next(caption_model.parameters()).dtype
                florence_inputs = {k: v.to(device, model_dtype) if torch.is_floating_point(v) else v.to(device) for k, v in florence_inputs.items()}
                
                with torch.no_grad():
                    florence_generated_ids = caption_model.generate(
                        **florence_inputs,
                        max_new_tokens=256,
                        num_beams=beam_size,
                        do_sample=False,
                        use_cache=False 
                    )
                
                florence_generated_texts = caption_processor.batch_decode(florence_generated_ids, skip_special_tokens=False)
                
                for i, text in enumerate(florence_generated_texts):
                    image_size = (batch_images[i].width, batch_images[i].height)
                    try:
                        parsed = caption_processor.post_process_generation(text, task=florence_prompt, image_size=image_size)
                        desc_text = parsed[florence_prompt]
                    except Exception as e:
                        desc_text = caption_processor.batch_decode([florence_generated_ids[i]], skip_special_tokens=True)[0]
                    
                    final_descriptions[i] = cleanup_text(desc_text)
                
                del florence_inputs, florence_generated_ids
                metrics.stop("Florence-2 Inference")

            elif caption_model_type == "ollama":
                print("  Generating descriptions via Ollama (Parallel)...")
                metrics.start("Ollama Description")
                from concurrent.futures import ThreadPoolExecutor
                
                def fetch_ollama_description(idx, img_path):
                    try:
                        factual_prompt = (
                            "Describe this image in detail. Be factual. No poetic flair. "
                            "Do not start with 'The image shows' or 'Here is a description'. "
                            "Provide ONLY the description itself. No markdown."
                        )
                        desc_text = describe_image(
                            img_path, 
                            model="gemma3:12b-it-qat", 
                            provider="ollama",
                            prompt=factual_prompt
                        )
                        return idx, cleanup_text(desc_text)
                    except Exception as e:
                        return idx, ""

                with ThreadPoolExecutor(max_workers=len(batch_paths)) as executor:
                    results = list(executor.map(lambda p: fetch_ollama_description(*p), enumerate(batch_paths)))
                    for idx, desc in results:
                        final_descriptions[idx] = desc
                metrics.stop("Ollama Description")

            # --- 3. Process Results ---
            print("  Upserting to database...")
            metrics.start("DB Upsert")
            for idx, image_path in enumerate(batch_paths):
                image = batch_images[idx]
                description_text = final_descriptions[idx] # Use unified description
                embedding = embeddings_np[idx]

                result = {
                    "image_path": image_path,
                    "captions": [], # Explicitly empty as per design decision to unify on description
                    "description": description_text,
                    "embedding": embedding,
                    "width": image.width,
                    "height": image.height,
                    "metadata": {
                        "env": detect_environment(),
                        "device": str(device),
                        "batch_size": batch_size,
                        "embedding_model": embedding_model_type,
                        "description_model": caption_model_type,
                    },
                }
                
                # Filter out images with empty descriptions to prevent empty embeddings
                if not description_text:
                    print(f"[WARNING] Skipping image {image_path} due to empty description.")
                    continue

                processed_count += 1

                if consumers:
                    for consumer in consumers:
                        try:
                            consumer(result)
                        except Exception as e:
                            traceback.print_exc()

                yield result
            metrics.stop("DB Upsert")
            metrics.stop("Batch Total Processing")
            
            # Final cleanup for the batch
            del batch_images
            del embeddings_np

        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            traceback.print_exc()
        
        # --- Memory Cleanup ---
        import gc
        gc.collect()
        if "mps" in str(device):
            torch.mps.empty_cache()
        elif "cuda" in str(device):
            torch.cuda.empty_cache()
        
        metrics.start("Image Fetching (Disk)")

    total_end_time = time.time() 
    metrics.stop("Image Fetching (Disk)") # Stop the last one which might be waiting for EOF
    print(f"\n--- Processing Summary ---")
    print(f"Processed {processed_count} images in {total_end_time - total_start_time:.2f} seconds.")
    metrics.print_summary()


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

# ------------------------------
# Example consumer
# ------------------------------
def print_consumer(data: Dict[str, Any]):
    """Simple consumer that prints output."""
    print("PRINT:", data["image_path"])
    print("  Caption:", [c["text"][:50] + "..." for c in data.get("captions", [])])
    print("  Description:", data.get("description", "")[:100] + "...")

# ------------------------------
# Persistence consumer
# ------------------------------
def make_persistence_consumer(collection_name: str = None, path: str = None, embedding_model_type: str = "clip"):
    """
    Returns a consumer that writes image, captions, description, metadata, and embedding
    into ChromaDB via PersistenceAPI.
    """
    api = PersistenceAPI(collection_name=collection_name, path=path, embedding_model_type=embedding_model_type)

    def _consumer(data: Dict[str, Any]):
        file_path = data["image_path"]
        emb = data["embedding"]
        captions = [c["text"] for c in data.get("captions", [])]
        description = data.get("description") # Get detailed description
        width = data.get("width")
        height = data.get("height")
        metadata = data.get("metadata", {})

        # Convert numpy embedding → Python list[float] for PersistenceAPI
        emb_list = emb.tolist() if hasattr(emb, "tolist") else emb

        api.insert_or_update(
            file_path=file_path,
            embedding=emb_list,
            captions=captions,
            description=description, # Pass description to API
            metadata=metadata,
            width=width,
            height=height,
        )

    return _consumer

# ------------------------------
# Usage
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Describe an image using Florence-2 or Hugging Face models.')
    parser.add_argument('--ip', required=True, dest='image_path', help='Path to the image file')
    parser.add_argument('--model', default='clip', choices=['clip', 'siglip'], help='Embedding model to use.')

    args = parser.parse_args()

    print("-START-")
    
    cache = "processed_cache.jsonl"
    # Add the persistence consumer alongside the print consumer
    persist_consumer = make_persistence_consumer(embedding_model_type=args.model)
    for _ in process_folder_parallel_caption(
        path=args.image_path,
        consumers=[print_consumer, persist_consumer],
        cache_file=cache,
        embedding_model_type=args.model
    ):
        pass
