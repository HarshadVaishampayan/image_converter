import argparse
import numpy as np
import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from PIL import Image
from persistence import normalize_embedding
from model_manager import ModelManager
from core.embeddings import EmbeddingGenerator
from core.faiss_search import FAISSSearch

import caption_image
import persistence

def search_similar(query_text: str = None, image_path: str = None, force_rebuild: bool = False, top_k: int = 6, sensitivity: float = 1.5) -> List[Dict]:
    """
    Search for similar images using either text query or image path.
    
    Args:
        query_text (str, optional): Text query to search with
        image_path (str, optional): Path to image to search with
        force_rebuild (bool): Whether to force rebuild the FAISS index
        top_k (int): The number of top matches to return.
        sensitivity (float): The score threshold to filter results.
        
    Returns:
        List[Dict]: List of results, each containing file_path, score, and captions
    """
    db = persistence.PersistenceAPI()
    if force_rebuild:
        print("Forcing FAISS index rebuild...")
        db._rebuild_index()
        db.save_index()

    embedding_generator = EmbeddingGenerator()
    faiss_search = FAISSSearch(index_path="faiss.index")

    if image_path:
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert('RGB')
        embedding = embedding_generator.generate_embedding(img, device="cpu")
    elif query_text:
        embedding = embedding_generator.generate_embedding(query_text, device="cpu")
    else:
        raise ValueError("Must provide either query_text or image_path")

    # Normalize embedding
    embedding = normalize_embedding(embedding)
    
    # Search for similar images
    results = faiss_search.find_similar_images(embedding, top_k=top_k)

    # Filter results based on sensitivity
    filtered_results = [r for r in results if r['score'] <= sensitivity]
    
    print(f"Found {len(results)} matches, returning {len(filtered_results)} after filtering with sensitivity {sensitivity}")
    
    return filtered_results

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on image or query string.")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS index")
    args = parser.parse_args()

    try:
        results = search_similar(
            query_text=args.query,
            image_path=args.image,
            force_rebuild=args.rebuild_index
        )
            
        if not results:
            print("No similar images found.")
            return

        for result in results:
            print(f"Path: {result['file_path']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Captions: {result['captions']}")
            print("-" * 20)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()