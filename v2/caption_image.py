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
import logging
import tagging

log = logging.getLogger("image_converter.index")

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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _get_image_paths(path: str, recursive: bool = True) -> List[str]:
    """Get image paths under `path` (all subfolders if recursive, else top level only)."""
    image_paths = []
    if os.path.isfile(path):
        if path.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for file in sorted(files):
                if file.lower().endswith(IMAGE_EXTENSIONS) and not file.startswith("."):
                    image_paths.append(os.path.join(root, file))
            if not recursive:
                break
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
            return image, image_path, None
        except Exception as e:
            # Keep the reason so the caller can report which file failed and why
            return None, image_path, f"{type(e).__name__}: {e}"


def collate_fn(batch):
    """Split a batch into loaded images and (path, reason) load failures."""
    loaded = [item for item in batch if item[0] is not None]
    failures = [(path, reason) for image, path, reason in batch if image is None]
    images = [item[0] for item in loaded]
    paths = [item[1] for item in loaded]
    return images, paths, failures


class FolderUnavailable(OSError):
    """The folder being indexed disappeared mid-run (e.g. a network share dropped)."""


# ------------------------------
# Main Processing Function
# ------------------------------
def process_folder_parallel_caption(
    path: str,
    consumers: List[Callable[[Dict[str, Any]], None]] = None,
    cache_file: str = None,
    max_workers: int = None,
    embedding_model_type: str = "siglip",
    caption_model_type: str = "ollama",
    recursive: bool = True,
    image_paths: List[str] = None,
    on_issue: Callable[[str, str, str], None] = None,
) -> Generator:
    """
    Process images in a directory (recursively by default) using:
    1. Selected caption model (Florence-2 or Ollama) for description.
    2. Selected embedding model (CLIP/SigLIP) for vectors.

    Pass `image_paths` to process an explicit list instead of scanning `path`.
    Yields one result per image that was described and saved by every consumer.
    `on_issue(kind, path, reason)` is called for each image that "failed"
    (unreadable, model error, save error) or was "skipped" (empty description).
    A batch that errors is retried one image at a time, so one bad file only
    costs itself.
    """
    metrics = Metrics()
    metrics.reset()

    total_start_time = time.time()
    log.info("Indexing %s | embed=%s | captions=%s", path, embedding_model_type, caption_model_type)

    def report(kind, image_path, reason):
        (log.error if kind == "failed" else log.warning)("%s %s: %s", kind.upper(), image_path, reason)
        if on_issue:
            on_issue(kind, image_path, reason)

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
    paths_to_process = image_paths if image_paths is not None else _get_image_paths(path, recursive)
    if not paths_to_process:
        log.warning("No valid images found in %s", path)
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

    total_images = len(paths_to_process)
    log.info("Found %d images; batch size %d", total_images, batch_size)

    processed_count = 0
    attempted = 0

    # Task Prompts
    florence_prompt = "<MORE_DETAILED_CAPTION>"
    factual_prompt = (
        "Describe this image in detail. Be factual. No poetic flair. "
        "Do not start with 'The image shows' or 'Here is a description'. "
        "Provide ONLY the description itself. No markdown."
    )

    def describe_batch(batch_images, batch_paths):
        """Embeddings + tags + descriptions for one batch. Raises on model errors."""
        # --- 1. Embedding Inference (Universal) ---
        with torch.no_grad():
            metrics.start("Image Embedding")
            embeddings = embedding_model.encode(batch_images, convert_to_tensor=True, device=device)
            metrics.stop("Image Embedding")
        embeddings_np = embeddings.float().cpu().numpy()
        del embeddings

        # --- 1b. Zero-shot tags (SigLIP only): specific labels + their categories ---
        batch_tags = tagging.tag_embeddings(embedding_model, embeddings_np)

        # --- 2. Image Description (Model Specific) ---
        descriptions = ["" for _ in batch_paths]
        why_empty = ["captioner returned an empty description" for _ in batch_paths]

        if caption_model_type == "florence2":
            metrics.start("Florence-2 Inference")
            florence_inputs = caption_processor(text=[florence_prompt] * len(batch_images),
                                                images=batch_images, return_tensors="pt")
            # Ensure inputs match model's precision
            model_dtype = next(caption_model.parameters()).dtype
            florence_inputs = {k: v.to(device, model_dtype) if torch.is_floating_point(v) else v.to(device)
                               for k, v in florence_inputs.items()}
            with torch.no_grad():
                generated_ids = caption_model.generate(
                    **florence_inputs,
                    max_new_tokens=256,
                    num_beams=beam_size,
                    do_sample=False,
                    use_cache=False,
                )
            generated_texts = caption_processor.batch_decode(generated_ids, skip_special_tokens=False)
            for i, text in enumerate(generated_texts):
                image_size = (batch_images[i].width, batch_images[i].height)
                try:
                    parsed = caption_processor.post_process_generation(text, task=florence_prompt, image_size=image_size)
                    desc_text = parsed[florence_prompt]
                except Exception:
                    desc_text = caption_processor.batch_decode([generated_ids[i]], skip_special_tokens=True)[0]
                descriptions[i] = cleanup_text(desc_text)
            del florence_inputs, generated_ids
            metrics.stop("Florence-2 Inference")

        elif caption_model_type == "ollama":
            metrics.start("Ollama Description")
            from concurrent.futures import ThreadPoolExecutor

            def fetch_ollama_description(idx, img_path):
                try:
                    desc_text = describe_image(img_path, model="gemma3:12b-it-qat",
                                               provider="ollama", prompt=factual_prompt)
                    return idx, cleanup_text(desc_text), None
                except Exception as e:
                    return idx, "", f"Ollama error: {type(e).__name__}: {e}"

            with ThreadPoolExecutor(max_workers=len(batch_paths)) as executor:
                for idx, desc, err in executor.map(lambda p: fetch_ollama_description(*p), enumerate(batch_paths)):
                    descriptions[idx] = desc
                    if err:
                        why_empty[idx] = err
            metrics.stop("Ollama Description")

        results = []
        for idx, image_path in enumerate(batch_paths):
            image = batch_images[idx]
            results.append({
                "image_path": image_path,
                "captions": [],  # Explicitly empty as per design decision to unify on description
                "description": descriptions[idx],
                "embedding": embeddings_np[idx],
                "width": image.width,
                "height": image.height,
                "metadata": {
                    "env": detect_environment(),
                    "device": str(device),
                    "batch_size": batch_size,
                    "embedding_model": embedding_model_type,
                    "description_model": caption_model_type,
                    "tags": json.dumps(batch_tags[idx]),
                },
                "_why_empty": why_empty[idx],
            })
        return results

    def folder_gone():
        return not os.path.exists(path)

    metrics.start("Image Fetching (Disk)")
    for batch_images, batch_paths, load_failures in data_loader:
        metrics.stop("Image Fetching (Disk)")
        attempted += len(batch_paths) + len(load_failures)

        if load_failures and folder_gone():
            raise FolderUnavailable(
                f"The folder is no longer accessible: {path} (was a drive or network share disconnected?)")
        for image_path, reason in load_failures:
            report("failed", image_path, f"could not read image: {reason}")

        if batch_images:
            log.info("Images %d-%d of %d", attempted - len(batch_paths) - len(load_failures) + 1,
                     attempted, total_images)
            metrics.start("Batch Total Processing")
            try:
                results = describe_batch(batch_images, batch_paths)
            except Exception:
                log.exception("Batch failed; retrying its %d image(s) one at a time", len(batch_paths))
                results = []
                for image, image_path in zip(batch_images, batch_paths):
                    try:
                        results.extend(describe_batch([image], [image_path]))
                    except Exception as e:
                        report("failed", image_path, f"model error: {type(e).__name__}: {e}")

            # --- 3. Save + yield ---
            metrics.start("DB Upsert")
            for result in results:
                image_path = result["image_path"]
                why_empty = result.pop("_why_empty")
                # Filter out images with empty descriptions to prevent empty embeddings
                if not result["description"]:
                    report("skipped", image_path, why_empty)
                    continue
                try:
                    for consumer in consumers or []:
                        consumer(result)
                except Exception as e:
                    log.exception("Saving %s failed", image_path)
                    report("failed", image_path, f"could not save to index: {type(e).__name__}: {e}")
                    continue
                processed_count += 1
                yield result
            metrics.stop("DB Upsert")
            metrics.stop("Batch Total Processing")
            del batch_images, results

        # --- Memory Cleanup ---
        import gc
        gc.collect()
        if "mps" in str(device):
            torch.mps.empty_cache()
        elif "cuda" in str(device):
            torch.cuda.empty_cache()

        metrics.start("Image Fetching (Disk)")

    total_end_time = time.time()
    metrics.stop("Image Fetching (Disk)")  # Stop the last one which might be waiting for EOF
    log.info("Indexed %d of %d images in %.1fs", processed_count, total_images, total_end_time - total_start_time)
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
