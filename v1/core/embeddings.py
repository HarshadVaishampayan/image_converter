from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np
from PIL import Image

class EmbeddingGenerator:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, input_data: Union[str, Image.Image], device: str = "cpu") -> np.ndarray:
        """
        Generate an embedding for the given input data.

        Args:
            input_data (Union[str, Image.Image]): The input data, either a text string or an image.
            device (str): The device to run the model on (e.g., "cpu" or "cuda").

        Returns:
            np.ndarray: The generated embedding.
        """
        print(f"Input type: {type(input_data)}")
        print(f"Model name: {self.model}")
        if isinstance(input_data, str):
            print(f"Text input: {input_data}")
            embedding = self.model.encode([input_data], convert_to_tensor=True, device=device)
            embedding = embedding[0]  # Remove batch dimension for text inputs
            print(f"Generated embedding shape: {embedding.shape}")
        elif isinstance(input_data, Image.Image):
            print(f"Image input size: {input_data.size}")
            embedding = self.model.encode(input_data, convert_to_tensor=True, device=device)
            print(f"Generated embedding shape: {embedding.shape}")
        else:
            raise ValueError("Unsupported input type. Must be a string or PIL.Image.Image.")

        return embedding.cpu().numpy()