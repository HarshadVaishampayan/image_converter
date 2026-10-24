# image_converter

## Setup: Virtual Environment & Installing Packages

1. **Create a virtual environment** (Python 3.10 recommended for GPU support):
	```sh
	python3.10 -m venv .venv310
	source .venv310/bin/activate
	```
	Or for CPU-only / Python 3.12:
	```sh
	python3.12 -m venv .venv
	source .venv/bin/activate
	```

2. **Install dependencies**
	```sh
	pip install --upgrade pip
	pip install -r requirements.txt   # one file for macOS (MPS), Linux (CUDA) and CPU; Python 3.12–3.14
	```

3. **(Optional, for GPU/CUDA users)**
	If using Python 3.10 and an NVIDIA GPU, install PyTorch with CUDA 11.7 support:
	```sh
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
	```


## How to Use

### 1. Image Conversion
Convert and resize images using the CLI:


```sh
python run_convert_image.py --ip <input_image> [--op <output_image>] --w <width> --f <format> [--h <height>] [--c <color>]
```

**Parameters:**
- `--ip` (required): Path to the input image
- `--op`: Path to save the converted image (optional; default: n_<input_image> in same folder)
- `--w` (required): Width of the output image
- `--h`: Height of the output image (optional; if not set, height is scaled proportionally)
- `--f` (required): Output format (`jpeg` or `png`)
- `--c`: Color mode (`greyscale` or `rgb`, optional)

**Example:**
```sh
python run_convert_image.py --ip input.jpg [--op output.png] --w 256 --f png [--h 128] [--c greyscale]
```

### 2. Image Description (Ollama or Hugging Face)
Describe an image using a local Ollama model or Hugging Face pipeline:


```sh
python run_describe_image.py --ip <image_path> [--m <model_name>] [--provider <ollama|huggingface>] [--device <auto|cpu|cuda>]
```

**Parameters:**
- `--ip` (required): Path to the image file
- `--m`: Model to use (default: `llava:7b-v1.5-q4_1`)
- `--provider`: Provider to use: `ollama` (default) or `huggingface`
- `--device`: Device for Hugging Face: `auto` (default), `cpu`, or `cuda`

**Example (single file):**
```sh
python run_describe_image.py --ip /path/to/image.jpg [--m Salesforce/blip-image-captioning-base] [--provider huggingface] [--device cpu]
```

**Example (all files in a directory):**
```sh
for img in /path/to/images/*.[jJpP][pPnNgG]*; do python run_describe_image.py --ip "$img" [--m Salesforce/blip-image-captioning-base] [--provider huggingface] [--device cpu]; done
```

---

## Python Files Overview

- **convert_image.py**: Core logic for converting and resizing images. Accepts a parameter map, supports color mode, format, and EXIF handling.
- **run_convert_image.py**: Command-line interface for image conversion. Parses arguments and calls `convert_image`.
- **describe_image.py**: Describes images using either a local Ollama model or Hugging Face pipeline (API/local). Supports provider/model/device selection and robust error handling.
- **run_describe_image.py**: Command-line interface for image description. Handles batch processing, logging, timing, and calls `describe_image` with all CLI parameters.
- **test_convert_image.py**: Unit tests for image conversion logic, covering resizing, color, format, and error cases.
- **requirements.txt**: Single dependency list for all platforms (torch 2.14, transformers 5, sentence-transformers 6, chromadb 1.5, numpy 2).
- **duplicates/**: Flask blueprint for the Duplicates page (Level 1: name + size; Level 2: exact bytes or image similarity via `ModelManager`).
- **.gitignore**: Standard Python/Venv ignores.