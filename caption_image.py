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
import traceback
import argparse

# ------------------------------
# Process in batches
# ------------------------------
def process_folder_parallel_blip(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = 4
) -> Generator:
    print(f"\nProcessing path: {path}")
    
    # Get cached models and parameters
    model_manager = ModelManager()
    processor, model, beam_size, batch_size = model_manager.get_blip_models()
    clip_model = model_manager.get_clip_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if path is a file or directory
    if os.path.isfile(path):
        paths_to_process = [path]
    elif os.path.isdir(path):
        paths_to_process = [
            os.path.join(path, p) for p in os.listdir(path)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        print(f"Error: {path} is not a valid file or directory")
        return

    for image_path in paths_to_process:
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Generate captions using BLIP
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, num_beams=beam_size, 
                                  num_return_sequences=beam_size, max_new_tokens=100)
            captions = [processor.decode(output, skip_special_tokens=True) 
                       for output in outputs]

            # Generate embedding using CLIP
            embedding = clip_model.encode([image], convert_to_tensor=True, device=device)
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

            # Notify consumers safely
            if consumers:
                for consumer in consumers:
                    try:
                        consumer(result)
                    except Exception as e:
                        print(f"[WARNING] Consumer {consumer.__name__} failed for {image_path}: {e}")
                        traceback.print_exc()

            # Update cache
            if cache_file:
                with open(cache_file, "a") as f:
                    json.dump({
                        "image_path": result["image_path"],
                        "captions": result["captions"]
                    }, f)
                    f.write("\n")

            yield result

        except Exception as e:
            print(f"[ERROR] Processing {image_path} failed: {e}")
            traceback.print_exc()

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
        max_workers=4
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

