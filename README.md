# Image Converter — Image processing, description, and semantic search

A Python toolkit for converting images, generating rich natural-language descriptions, producing embeddings, and performing semantic/hybrid image search. The repository contains two versions (`v1` for historical reference and `v2` for the current implementation). `v2` is the actively developed code path and uses ChromaDB for persistent vector storage.

This README covers: a feature guide, quick setup, usage examples (CLI and web), developer notes, and troubleshooting tips.

---

## Quick highlights

- Image conversion and format conversion (resize, color conversion).
- Image description using local Ollama models or Hugging Face models (Florence-2).
- Embeddings produced with SigLIP or CLIP; embeddings are stored and queried with ChromaDB (v2) or FAISS+SQLite (v1).
- Hybrid search combines visual similarity and text relevance (caption/description re-ranking).
- CLI and Flask web UI for interactive use.

---

## Feature guide (what the project provides)

1. Image conversion
   - Resize, convert format (JPEG/PNG), and change color mode (grayscale/RGB) via `convert_image.py` (present in both `v1` and `v2`).

2. Image description / captioning
   - Generate detailed descriptions using local Ollama models (default in `v2/describe_image.py`) or Hugging Face pipelines (when installed).
   - The `caption_image` pipeline orchestrates description generation and embedding extraction for bulk ingestion.

3. Embeddings and vector storage
   - `v2` uses ChromaDB (persistent client) and exposes `PersistenceAPI` for upsert/query/cleanup.
   - `v1` provides an alternate persistence implementation using SQLite + FAISS (images.db and faiss.index).
   - Embeddings can come from `siglip` or `clip` (configurable).

4. Hybrid semantic search
   - Search by text query, example image, or both. Results combine nearest-neighbor embedding distance with text-based re-ranking.

5. Interfaces
   - `v2/image_cli.py`: Click-based CLI exposing ingestion, search, inspection, cleanup, stats, and web server startup.
   - `v2/gui.py`: Flask app used by the `web` CLI command to display search results in a grid.

6. Model & resource management
   - `v2/model_manager.py` centralizes model loading and device detection (CUDA / MPS / CPU). Use environment variables such as `HF_TOKEN` for private HF models.

7. Metrics & observability
   - Lightweight timing via `v2/metrics.py` to track ingestion and embedding costs.

8. Tests & examples
   - Unit tests exist in `v1/tests` and `v2/tests` for core components (embeddings, search). Use `pytest` to run them.

---

## How-to guide (setup and common workflows)

Prerequisites

- Python 3.10+ recommended. Use matching `requirements*.txt` for environment-specific packages.
- Optional: A machine with CUDA for GPU acceleration, or macOS with MPS support for Apple Silicon.
- Optional: Ollama installed and running locally (if you want local LLM image descriptions). `v2/describe_image.py` calls `http://localhost:11434/api/generate` by default.
- (Optional) A Hugging Face token (`HF_TOKEN`) if using private or rate-limited HF models.

Basic setup

1. Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd image_converter
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies for the `v2` codebase (recommended):

```bash
cd v2
pip install --upgrade pip
pip install -r requirements.txt
```

Notes on dependency files

- `requirements.txt`, `requirements_mac.txt`, `requirements310.txt`, and `requirements312.txt` exist to support different targets. Prefer `v2/requirements.txt` for the primary v2 development path.

Configure model credentials (optional)

```bash
# Set Hugging Face token for private models
export HF_TOKEN="<your-hf-token>"

# Ensure Ollama (if used) is running locally on port 11434
# ollama serve  # (or follow Ollama docs)
```

Common workflows

Index (describe) a folder of images (ingest + index):

```bash
# from repo root
python v2/image_cli.py describe --ip /path/to/images --embed-model siglip --cap-model ollama --chroma-path ./chroma_db
```

Search (text or image query):

```bash
# text query
python v2/image_cli.py search --query "a dog running on the beach" --embed-model siglip

# image query
python v2/image_cli.py search --image /path/to/query.jpg --embed-model siglip
```

Inspect a stored image's metadata and embeddings:

```bash
python v2/image_cli.py inspect --image-path /path/to/image.jpg --embed-model siglip
```

Run the web UI (Flask):

```bash
python v2/image_cli.py web --host 0.0.0.0 --port 5000
# Then open http://localhost:5000
```

Cleanup empty text embeddings (captions/description chunks):

```bash
python v2/image_cli.py cleanup-embeddings --embed-model siglip
```

Purge a collection (destructive):

```bash
python v2/image_cli.py purge --model siglip
```

Developer & maintenance commands

- Run unit tests:

```bash
pytest -q
```

- Format and lint (recommended):

```bash
pip install black ruff mypy
black .
ruff check .
mypy v2 --ignore-missing-imports
```

---

## Developer notes (where to look)

- `v2/image_cli.py` — CLI front-end (Click) for v2 workflows.
- `v2/caption_image.py` — Orchestrates description generation and embedding for ingestion.
- `v2/persistence.py` — ChromaDB-backed PersistenceAPI (insert/update/query/upsert/cleanup).
- `v2/model_manager.py` — Loads and caches models (embedding and captioning).
- `v2/find_similar.py` — Hybrid search logic combining embeddings and text signals.
- `v2/gui.py` — Flask templates and handlers used by the web UI.
- `v1/` — Historical v1 implementation using SQLite + FAISS; useful reference for alternative persistence patterns.

Design considerations

- `v2` favors ChromaDB for persistent vector store and richer metadata handling.
- Model loading is centralized to avoid repeated heavyweight downloads and to manage device placement (CUDA/MPS/CPU).
- Persistence stores image entries plus caption and description chunk embeddings to support hybrid search and fine-grained re-ranking.

---

## Troubleshooting & tips

- If descriptions are missing or fail, ensure Ollama is reachable at `http://localhost:11434` or switch to `--cap-model florence2` (requires HF models installed and `HF_TOKEN` if private).
- Monitor memory when indexing large image sets; consider batching and smaller batch sizes for devices with limited RAM.
- If ChromaDB queries return nothing, confirm that the collection path (default `./chroma_db`) contains data and that you used the same `--embed-model` when indexing and searching.

---

## Contributing

Contributions are welcome. Suggested steps:

1. Fork the repository and create a feature branch.
2. Run formatting/linting and tests locally.
3. Open a PR with a clear description and tests for any behavioral changes.

---

If you'd like, I can also:

- Add a `pyproject.toml` and a `requirements-dev.txt` for development tools.
- Add a minimal GitHub Actions workflow to run linters and tests on PRs.
- Run an automatic code-format pass (`black`) and replace ad-hoc `print()` logging with structured `logging` in key modules.

---
