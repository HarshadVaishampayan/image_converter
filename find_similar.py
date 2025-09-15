import argparse
import numpy as np
import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from PIL import Image
from persistence import normalize_embedding
from model_manager import ModelManager

import caption_image
import persistence

def search_similar(query_text: str = None, image_path: str = None, force_rebuild: bool = False) -> List[Dict]:
    """
    Search for similar images using either text query or image path.
    
    Args:
        query_text (str, optional): Text query to search with
        image_path (str, optional): Path to image to search with
        force_rebuild (bool): Whether to force rebuild the FAISS index
        
    Returns:
        List[Dict]: List of results, each containing file_path, score, and captions
    """
    db = persistence.PersistenceAPI()
    model_manager = ModelManager()
    clip_model = model_manager.get_clip_model()
    device = model_manager.device

    if force_rebuild:
        print("Forcing FAISS index rebuild...")
        db._rebuild_index()
        db.save_index()
        return

    elif image_path:
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert('RGB')
        embedding = clip_model.encode(img, convert_to_tensor=True, device=device)
        embedding = embedding.cpu().numpy()
    
    elif query_text:
        embedding = clip_model.encode([query_text], convert_to_tensor=True, device=device)
        embedding = embedding[0].cpu().numpy()
    else:
        raise ValueError("Must provide either query_text or image_path")

    # Normalize embedding
    embedding = normalize_embedding(embedding)
    
    # Search for similar images
    results = db.find_similar_images(embedding)
    return results

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on image or query string.")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS index")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive:
        while True:
            try:
                print("\nEnter 'i' for image search, 'q' for text query, or 'exit' to quit")
                choice = input("> ").strip().lower()
                
                if choice == 'exit':
                    break
                elif choice == 'i':
                    path = input("Enter image path: ").strip()
                    if not os.path.exists(path):
                        print("Invalid path")
                        continue
                    img = Image.open(path).convert('RGB')
                    embedding = clip_model.encode(img, convert_to_tensor=True).cpu().numpy()
                elif choice == 'q':
                    query = input("Enter text query: ").strip()
                    embedding = clip_model.encode([query], convert_to_tensor=True)[0].cpu().numpy()
                else:
                    print("Invalid choice")
                    continue

                results = db.find_similar_images(embedding)
                print("\nResults:")
                for r in results:
                    print(f"Path: {r['file_path']}")
                    print(f"Score: {r['score']:.4f}")
                    print(f"Captions: {r['captions']}")
                    print("-" * 20)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
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