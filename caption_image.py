from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    Blip2ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Dict, Any, Generator
from persistence import PersistenceAPI
from model_manager import ModelManager, detect_environment, load_models
import torch
import json
import os
import time
import traceback
import argparse
import concurrent.futures

# ------------------------------
# Process in batches
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

def process_folder_parallel_blip(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = None,
) -> Generator:
    """Process images in a directory recursively using BLIP for captioning."""
    print(f"\nProcessing path: {path}")
    start_time = time.time()
    processed_count = 0

    if max_workers is None:
        max_workers = os.cpu_count()
        print(f"Using default max_workers: {max_workers}")

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

    print(f"Found {len(paths_to_process)} images to process with {max_workers} workers")

    def process_image(image_path):
        try:
            print(f"Processing: {image_path}")
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Generate captions using BLIP
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs, num_beams=beam_size, num_return_sequences=beam_size, max_new_tokens=100
            )
            captions = [
                processor.decode(output, skip_special_tokens=True) for output in outputs
            ]

            # Generate embedding using CLIP
            embedding = clip_model.encode(
                [image], convert_to_tensor=True, device=device
            )
            embedding = embedding[0].cpu().numpy()  # Convert to numpy array

            # Create result dictionary with embedding
            result = {
                "image_path": image_path,
                "captions": [{"text": cap} for cap in captions],
                "embedding": embedding,  # Add embedding to result
                "width": image.width,
                "height": image.height,
                "metadata": {
                    "env": detect_environment(),
                    "device": str(device),
                    "batch_size": batch_size,
                    "beam_size": beam_size,
                },
            }
            return result

        except Exception as e:
            print(f"[ERROR] Processing {image_path} failed: {e}")
            traceback.print_exc()
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(process_image, path): path for path in paths_to_process
        }

        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result:
                processed_count += 1
                image_path = result["image_path"]
                # Notify consumers safely
                if consumers:
                    for consumer in consumers:
                        try:
                            consumer(result)
                        except Exception as e:
                            print(
                                f"[WARNING] Consumer {consumer.__name__} failed for {image_path}: {e}"
                            )
                            traceback.print_exc()

                # Update cache
                if cache_file:
                    with open(cache_file, "a") as f:
                        json.dump(
                            {
                                "image_path": result["image_path"],
                                "captions": result["captions"],
                            },
                            f,
                        )
                        f.write("\n")

                yield result
    
    end_time = time.time()
    print(f"Processed {processed_count} images in {end_time - start_time:.2f} seconds.")


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

