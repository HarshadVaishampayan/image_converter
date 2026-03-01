import unittest
import numpy as np
from PIL import Image
from core.embeddings import EmbeddingGenerator

class TestEmbeddingGenerator(unittest.TestCase):

    def setUp(self):
        self.embedding_generator = EmbeddingGenerator()

    def test_generate_embedding_from_text(self):
        text = "A sample text for embedding generation."
        embedding = self.embedding_generator.generate_embedding(text)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], 512)  # Assuming the embedding size is 512

    def test_generate_embedding_from_image(self):
        image = Image.new('RGB', (224, 224), color='red')
        embedding = self.embedding_generator.generate_embedding(image)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], 512)  # Assuming the embedding size is 512

    def test_generate_embedding_invalid_input(self):
        with self.assertRaises(ValueError):
            self.embedding_generator.generate_embedding(12345)  # Invalid input type

if __name__ == "__main__":
    unittest.main()