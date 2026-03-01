import unittest
import numpy as np
import faiss
from core.faiss_search import FAISSSearch

class TestFAISSSearch(unittest.TestCase):

    def setUp(self):
        # Create a temporary FAISS index for testing
        self.index_path = "test_faiss.index"
        self.dimension = 512
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.write_index(self.index, self.index_path)
        self.faiss_search = FAISSSearch(index_path=self.index_path)

    def tearDown(self):
        # Clean up the temporary FAISS index
        import os
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    def test_find_similar_images(self):
        # Add dummy data to the index
        embeddings = np.random.rand(10, self.dimension).astype('float32')
        print(f"Adding embeddings to FAISS index: {embeddings.shape}")
        self.index.add(embeddings)
        print(f"Number of vectors in index after adding: {self.index.ntotal}")
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index written to: {self.index_path}")

        # Query the index
        query_embedding = embeddings[0]
        results = self.faiss_search.find_similar_images(query_embedding, top_k=5)

        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("file_path", result)
            self.assertIn("score", result)

    def test_empty_index(self):
        # Query an empty index
        query_embedding = np.random.rand(self.dimension).astype('float32')
        results = self.faiss_search.find_similar_images(query_embedding, top_k=5)

        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()