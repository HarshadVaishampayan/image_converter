import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoProcessor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Dict, Any, Generator
from persistence import PersistenceAPI;
import json, os, traceback, platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# ------------------------------
# Environment Detection
# ------------------------------
def detect_environment() -> str:
    """Detect if running on desktop or mobile (very basic heuristic)."""
    system = platform.system().lower()
    if system in ["linux", "windows", "darwin"]:
        if "ANDROID_ROOT" in os.environ:  # Android on Linux
            return "mobile"
        return "desktop"
    elif system in ["ios"]:
        return "mobile"
    else:
        return "desktop"

# ------------------------------
# Load Models Based on Environment
# ------------------------------
def load_models(env: str):
    """Load BLIP + CLIP models and config depending on environment type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if env == "desktop":
        blip_name = "Salesforce/blip-image-captioning-large"
        clip_name = "clip-ViT-B-32"
        batch_size = 2
        beam_size = 2
    else:
        blip_name = "Salesforce/blip-image-captioning-small"
        clip_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        batch_size = 2
        beam_size = 2

    return blip_name, clip_name, batch_size, beam_size, device

# ------------------------------
# Process in batches
# ------------------------------
def process_folder_parallel_blip(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = 4
) -> Generator:
    """Process a single image or all images in a folder using BLIP for captioning."""
    print(f"\nProcessing path: {path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load both BLIP and CLIP models
    blip_name, clip_name, batch_size, beam_size, _ = load_models(detect_environment())
    processor = BlipProcessor.from_pretrained(blip_name)
    model = BlipForConditionalGeneration.from_pretrained(blip_name).to(device)
    clip_model = SentenceTransformer(clip_name).to(device)  # Add CLIP model

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
    """
    Generate a caption using the BLIP-2 model fine-tuned on COCO.
    Localized function so it doesn't interfere with the existing BLIP pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b-coco", 
        device_map="auto"
    ).to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return caption

