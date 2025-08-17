# image_converter

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
- **requirements.txt / requirements310.txt / requirements312.txt**: Dependency lists for different Python versions and environments (see comments in each file).
- **.gitignore**: Standard Python/Venv ignores.