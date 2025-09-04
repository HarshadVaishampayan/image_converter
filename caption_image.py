import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, InstructBlipProcessor, InstructBlipForConditionalGeneration
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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

    return blip_name, clip_name, batch_size, beam_size

# ------------------------------
# Process in batches
# ------------------------------
def process_folder_parallel_blip(
    folder_path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = 4,
):
    """Process images in a folder in parallel using BLIP for captioning."""
    blip_name, clip_name, batch_size, beam_size = load_models(detect_environment())
    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_name)
    clip_model = SentenceTransformer(clip_name)
    device = get_device()
    blip_model.to(device)
    clip_model.to(device)

    images = []
    valid_paths = []

    # Load images
    for p in os.listdir(folder_path):
        full_path = os.path.join(folder_path, p)
        if os.path.isfile(full_path):
            try:
                img = Image.open(full_path).convert("RGB")
                images.append(img)
                valid_paths.append(full_path)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"[WARNING] Cannot open image {full_path}: {e}")
                continue

    if not images:
        return

    # Process in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_paths = valid_paths[i:i+batch_size]

        # --- GPU batch CLIP embeddings ---
        batch_embeddings = create_embeddings(clip_model, device, images)

        # --- Parallel BLIP captioning ---
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(blip_caption_single, img, blip_processor, blip_model, beam_size, device): idx
                for idx, img in enumerate(batch_images)
            }
            for future in as_completed(future_to_path):
                idx = future_to_path[future]
                img_path = batch_paths[idx]
                try:
                    captions = future.result()
                    if not captions:
                        continue

                    # Cosine similarity & top-N
                    image_emb = batch_embeddings[idx].unsqueeze(0)
                    caption_embs = clip_model.encode(captions, convert_to_tensor=True, device=device)
                    cosine_sim = torch.nn.functional.cosine_similarity(image_emb, caption_embs)
                    ranked = sorted(zip(captions, cosine_sim.tolist()), key=lambda x: x[1], reverse=True)
                    top_captions = ranked[:beam_size]
                    for cap, sim in top_captions:
                        print(f"[INFO] {img_path} → {cap} (sim={sim:.4f})")

                    result = {
                        "image_path": img_path,
                        "embedding": batch_embeddings[idx].cpu().numpy(),
                        "captions": [{"text": cap, "similarity": sim} for cap, sim in top_captions],
                        "width": batch_images[idx].width,
                        "height": batch_images[idx].height,
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
                                print(f"[WARNING] Consumer {consumer.__name__} failed for {img_path}: {e}")
                                traceback.print_exc()

                    # Update cache
                    if cache_file:
                        with open(cache_file, "a") as f:
                            json.dump({
                                "image_path": result["image_path"],
                                "embedding": result["embedding"].tolist(),
                                "captions": result["captions"]
                            }, f)
                            f.write("\n")

                    yield result

                except Exception as e:
                    print(f"[ERROR] BLIP captioning failed for {img_path}: {e}")
                    traceback.print_exc()

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_embeddings(clip_model, device, images):
    batch_embeddings = clip_model.encode(images, convert_to_tensor=True, device=device)
    return batch_embeddings

def blip_caption_single(img: Image.Image, blip_processor: BlipProcessor, blip_model: BlipForConditionalGeneration, beam_size: int, device: torch.device):
    """Generate captions for a single image using BLIP."""
    inputs = blip_processor(img, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, num_beams=beam_size, num_return_sequences=beam_size, max_new_tokens=100)
    captions = [blip_processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# ------------------------------
# Example consumer
# ------------------------------
def print_consumer(data: Dict[str, Any]):
    """Simple consumer that prints output."""
    print("PRINT:", data["image_path"], "→", [c["text"] for c in data["captions"]])

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
        folder_path=args.image_path,
        consumers=[print_consumer, persist_consumer],
        cache_file=cache,
        max_workers=4
    ):
        pass


# ------------------------------
# ⚡ Potential Improvements (Future Work)
# ------------------------------
# 1. Use asyncio + aiofiles for async image loading (avoid blocking on I/O).
# 2. Add dynamic GPU memory monitoring to auto-tune batch size.
# 3. Extend to support video frames (extract frames → run same embedding+caption pipeline).
# 4. Add better cache indexing (SQLite or DuckDB) for faster lookups instead of JSONL.
# 5. Add retry logic for flaky consumers (e.g. DB/network writers).
# 6. Add streaming/real-time processing support (watch a folder for new images).
# 7. Add support for ONNX/TensorRT model export for lighter mobile inference.

