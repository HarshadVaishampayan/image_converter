
import requests
import base64
import os
import traceback

def describe_image(image_path, model="llava:7b-v1.5-q4_1", provider="ollama", device="auto", prompt=None):
    """
    Describe an image using either Ollama (local) or Hugging Face (local) based on provider.
    provider: 'ollama' (default) or 'huggingface'
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if provider == "ollama":
        if not prompt:
            prompt = "Describe the image in as much detail as possible, including objects, scene, and quality."
        with open(image_path, 'rb') as img_file:
            img_bytes = img_file.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64]
        }
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        result = ''
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = line.decode('utf-8')
                import json
                obj = json.loads(data)
                result += obj.get('response', '')
            except Exception:
                continue
        return result
    elif provider == "huggingface":
        try:
            from transformers import pipeline
            import torch
            # Determine device
            if device == "cpu":
                device_arg = -1
            elif device == "cuda":
                device_arg = 0 if torch.cuda.is_available() else -1
            else:  # auto
                device_arg = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("image-to-text", model=model, device=device_arg)
            result = pipe(image_path)
            # result is a list of dicts with 'generated_text' or 'caption' keys
            if isinstance(result, list) and result:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                elif 'caption' in result[0]:
                    return result[0]['caption']
                else:
                    return str(result[0])
            return str(result)
        except ImportError:
            return "transformers library is not installed. Please install it to use Hugging Face provider."
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error running Hugging Face model: {e}\nTraceback:\n{tb}"
    else:
        return f"Unknown provider: {provider}"
