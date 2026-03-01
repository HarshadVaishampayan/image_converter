import faiss
import numpy as np
from typing import List, Dict

class FAISSSearch:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = self._load_index()

    def _load_index(self):
        """
        Load the FAISS index from the specified path.
        """
        try:
            return faiss.read_index(self.index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

    def find_similar_images(self, embedding: np.ndarray, top_k: int = 6) -> List[Dict]:
        """
        Find similar images using the FAISS index.

        Args:
            embedding (np.ndarray): The query embedding.
            top_k (int): The number of top matches to return.

        Returns:
            List[Dict]: List of results, each containing file_path and score.
        """
        distances, indices = self.index.search(embedding.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            results.append({"file_path": f"image_{idx}.jpg", "score": dist})
        return results