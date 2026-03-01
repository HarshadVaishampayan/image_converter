# Image Converter and Semantic Search (v2)

This project provides a robust command-line interface (CLI) and a Flask-based web application for image processing, detailed description generation, and semantic search. It leverages modern AI models for embeddings (SigLIP, CLIP) and descriptions (Gemma 3 via Ollama, Florence-2), and uses ChromaDB for persistent vector storage and retrieval.

The project has been refactored into a `v2` codebase, with the original `v1` components preserved for reference.

## Features

### Core Functionality
-   **Image Conversion**: Convert images to different formats, resize, and adjust color modes.
-   **Image Description**: Generate detailed descriptions for images using either local Ollama models (e.g., Gemma 3) or Hugging Face models (e.g., Florence-2).
-   **Semantic Search**: Find similar images based on text queries or example images.
-   **Hybrid Search**: Combines visual embedding similarity with keyword-based re-ranking for more accurate results.

### Technical Highlights
-   **Modular `v2` Architecture**: Clean separation of concerns with a dedicated `v2` directory.
-   **ChromaDB Integration**: Efficient local vector database for storing image embeddings, descriptions, and metadata.
-   **Dynamic Model Selection**: Choose between SigLIP and CLIP for embeddings, and Ollama (Gemma 3) or Florence-2 for descriptions via CLI flags.
-   **Performance Metrics**: Detailed timing and counting for various stages of the image ingestion pipeline.
-   **Web UI**: A Flask application (`gui.py`) providing a user-friendly interface for searching and browsing images.
-   **Data Management Utilities**: CLI commands for inspecting stored image data and cleaning up invalid entries.

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aidevvaishampayan/image_converter.git
cd image_converter
```

### 2. Create and Activate a Python Virtual Environment
It's highly recommended to use a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
_(Replace `python3` with `python3.10`, `python3.11`, or `python3.12` depending on your system and desired environment. Python 3.10 is often preferred for GPU setups.)_

### 3. Install Dependencies
Navigate into the `v2` directory and install the requirements.
```bash
cd v2
pip install --upgrade pip
pip install -r requirements.txt
# Or use specific requirements files based on your Python version and GPU availability:
# pip install -r requirements310.txt  # For Python 3.10 with potential CUDA support
# pip install -r requirements312.txt  # For Python 3.12 (CPU-only)
# pip install -r requirements_mac.txt # For macOS (MPS)
```
_Note: If you plan to use Ollama models, ensure [Ollama](https://ollama.ai/) is installed and running on your system._

## Usage

All primary functionalities are exposed via the `image_cli.py` script located in the `v2` directory.

```bash
cd v2 # Ensure you are in the v2 directory
python image_cli.py --help
```

### 1. Describe and Index Images

This command processes images, generates descriptions and embeddings, and stores them in ChromaDB.

```bash
python image_cli.py describe --ip <path/to/images> --embed-model <siglip|clip> --cap-model <ollama|florence2>
```
-   `--ip`: Path to an image file or a directory containing images.
-   `--embed-model`: Choose the embedding model (`siglip` or `clip`). Default is `siglip`.
-   `--cap-model`: Choose the description generation model (`ollama` or `florence2`). Default is `ollama`.

**Example:**
```bash
python image_cli.py describe --ip ~/Pictures/MyVacation --embed-model siglip --cap-model ollama
```

### 2. Search for Similar Images

Find images similar to a query text or another image.

```bash
python image_cli.py search [--query "search terms"] [--image /path/to/query_image.jpg] --embed-model <siglip|clip>
```
-   `--query`: Text query for semantic search.
-   `--image`: Path to an image whose embedding will be used for similarity search.
-   `--embed-model`: The embedding model used when indexing the images you want to search.

**Example (text query):**
```bash
python image_cli.py search --query "a cat playing with a ball" --embed-model siglip
```
**Example (image query):**
```bash
python image_cli.py search --image ~/Pictures/MyCat.jpg --embed-model siglip
```

### 3. Inspect Image Data

View all stored metadata, descriptions, and embeddings for a specific image.

```bash
python image_cli.py inspect --image-path /path/to/image.jpg --embed-model <siglip|clip>
```

### 4. Clean Up Empty Embeddings

Remove any ChromaDB entries that were created from empty captions or descriptions. Useful for cleaning up a database after previous indexing issues.

```bash
python image_cli.py cleanup-embeddings --embed-model <siglip|clip>
```

### 5. Run the Web Application

Start the Flask web server for a graphical interface.

```bash
python image_cli.py web [--host 0.0.0.0] [--port 5000] [--debug]
```

### 6. Purge Database

Delete an entire ChromaDB collection for a given embedding model.
```bash
python image_cli.py purge --model <siglip|clip>
```

## Project Structure (v2)

-   `v2/image_cli.py`: Main CLI entry point for all operations.
-   `v2/caption_image.py`: Handles image ingestion, description generation (Ollama/Florence-2), and embedding (SigLIP/CLIP).
-   `v2/persistence.py`: Manages ChromaDB interactions, including insert, update, retrieve, and cleanup operations.
-   `v2/model_manager.py`: Singleton for loading and managing AI models (embedding, description) efficiently.
-   `v2/find_similar.py`: Implements the hybrid search logic.
-   `v2/gui.py`: Flask application for the web interface.
-   `v2/metrics.py`: Singleton for tracking performance metrics during ingestion.
-   `v2/templates/` & `v2/static/`: Web UI templates and static assets.
-   `v2/requirements*.txt`: Python dependency lists.
-   `v1/`: Contains the original version of the project files.
-   `TODO/`: Markdown files outlining future development tasks.
-   `understanding.md`: Detailed internal document explaining v1 architecture and initial v2 goals.

## Development Roadmap (TODO)

-   **Refactor Caption Pipeline**: Further modularize `caption_image.py`.
-   **Dynamic Hybrid Search**: Add UI controls to configure search parameters (primary/secondary factors, influence).
-   **Synonym Expansion**: Implement query expansion using Ollama to improve text search recall.
-   **Asynchronous Image Pipelining**: Optimize ingestion by parallelizing Ollama descriptions with embedding/storage.
