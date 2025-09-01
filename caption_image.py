import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from typing import Callable, List, Dict, Any, Generator
import json, os, traceback, platform
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        blip_name = "Salesforce/blip-image-captioning-base"
        clip_name = "clip-ViT-B-32"
        batch_size = 8
        beam_size = 5
    else:
        blip_name = "Salesforce/blip-image-captioning-small"
        clip_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        batch_size = 2
        beam_size = 3

    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_name).to(device)
    clip_model = SentenceTransformer(clip_name)
    
    return blip_processor, blip_model, clip_model, batch_size, beam_size, device

# ------------------------------
# BLIP Captioning
# ------------------------------
def blip_caption_single(img_path: str, blip_processor, blip_model, beam_size: int, device) -> List[str]:
    """Generate multiple captions for a single image using BLIP."""
    try:
        raw_image = Image.open(img_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"[WARNING] Cannot open image {img_path}: {e}")
        return []

    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(
        **inputs, num_beams=beam_size, num_return_sequences=beam_size, max_new_tokens=30
    )
    return [blip_processor.decode(o, skip_special_tokens=True) for o in out]

# ------------------------------
# CLIP Embeddings (GPU batched)
# ------------------------------
def clip_embeddings(images: List[Image.Image], clip_model, device) -> torch.Tensor:
    """Compute CLIP embeddings for a batch of images."""
    return clip_model.encode(images, convert_to_tensor=True, device=device)

# ------------------------------
# Process folder with parallel BLIP
# ------------------------------
def process_folder_parallel_blip(
    folder_path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    extensions: List[str] = None,
    cache_file: str = None,
    max_workers: int = 4
) -> Generator[dict, None, None]:
    """
    Process all images in folder → embeddings + captions.
    Results are streamed to consumers + optional cache.
    """
    env = detect_environment()
    print(f"[INFO] Detected environment: {env}")
    blip_processor, blip_model, clip_model, batch_size, beam_size, device = load_models(env)

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    print(f"[INFO] Using batch_size={batch_size}, beam_size={beam_size}, device={device}")

    # Load cache
    processed_cache = set()
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_cache.add(entry["image_path"])
                except:
                    continue

    # Gather image paths
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                img_path = os.path.join(root, f)
                if img_path not in processed_cache:
                    image_paths.append(img_path)
                else:
                    print(f"[INFO] Skipping cached image: {img_path}")

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        valid_paths = []

        # Load images
        for p in batch_paths:
            try:
                print(f"[INFO] Loading image: {p}")
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"[WARNING] Cannot open image {p}: {e}")
                continue

        if not images:
            continue

        # --- GPU batch CLIP embeddings ---
        batch_embeddings = clip_embeddings(images, clip_model, device)

        # --- Parallel BLIP captioning ---
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(blip_caption_single, p, blip_processor, blip_model, beam_size, device): idx
                for idx, p in enumerate(valid_paths)
            }
            for future in as_completed(future_to_path):
                idx = future_to_path[future]
                img_path = valid_paths[idx]
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
                        "captions": [{"text": cap, "similarity": sim} for cap, sim in top_captions]
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

# ------------------------------
# Example consumer
# ------------------------------
def print_consumer(data: Dict[str, Any]):
    """Simple consumer that prints output."""
    print("PRINT:", data["image_path"], "→", [c["text"] for c in data["captions"]])

# ------------------------------
# Usage
# ------------------------------
if __name__ == "__main__":
    folder = "/backup/Hard Drive Backups/SSD 1/Aai Baba US visit"
    cache = "processed_cache.jsonl"
    for _ in process_folder_parallel_blip(
        folder_path=folder,
        consumers=[print_consumer],
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

