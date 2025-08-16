import requests
import base64
import os

def describe_image(image_path, model="llava:7b-v1.5-q4_1"):
    """
    Calls the Ollama API with the specified model to describe the image at the given path.
    Returns the model's response as a string.
    """
    prompt = "Describe the image in as much detail as possible, including objects, scene, and quality."

    # prompt = (
    #     "You are an image feature extractor. You excel at identifying objects, scenes, and image characteristics in an image. "
    #     "Identify the general scene, objects, and the theme of this image. Also, evaluate the image for clarity, blurriness, and other imperfections that degrade the image's quality. "
    #     "Return a structured output in the form of yaml. It should contain the following traits. "
    #     "Subjects: List the objects that are the primary subjects in the image. "
    #     "Scene: Describing the general setting of the image. "
    #     "Background: Any identified objects in the background. "
    #     "Theme: General sentiments of the overall subjects and scene, and "
    #     "Quality: List technical good an bad qualities of the image. "
    #     "Do not hallucinate. If you cannot read the image, say so outright."
    # )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

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
