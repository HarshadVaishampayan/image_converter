import argparse
import numpy as np
import os

from sentence_transformers import SentenceTransformer
import caption_image
import persistence
from PIL import Image
from persistence import normalize_embedding
from model_manager import ModelManager

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on image or query string.")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS index")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    db = persistence.PersistenceAPI()
    model_manager = ModelManager()
    clip_model = model_manager.get_clip_model()

    # Rebuild index if requested
    if args.rebuild_index:
        print("Forcing FAISS index rebuild...")
        db._rebuild_index()
        db.save_index()

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
        # Detect environment and load models
        env = caption_image.detect_environment()
        print(f"Running in environment: {env}")
        
        # Load models
        blip_name, clip_name, batch_size, beam_size, device = caption_image.load_models(env)
        clip_model = SentenceTransformer(clip_name)
        db = persistence.PersistenceAPI()
        
        # Optionally force rebuild index
        if args.rebuild_index:
            print("Forcing FAISS index rebuild...")
            db._rebuild_index()

        # Generate embedding based on input type
        if args.image:
            # Use same image encoding as during insertion
            img = Image.open(args.image).convert('RGB')
            embedding = clip_model.encode(img, convert_to_tensor=True, device=device)
            embedding = embedding.cpu().numpy()
        
        elif args.query:
            # For text queries, we need to use CLIP's text encoder
            # but ensure the embedding is normalized the same way
            text_embedding = clip_model.encode(args.query, convert_to_tensor=True, device=device)
            text_embedding = text_embedding.cpu().numpy()
            # Optional: Add additional normalization to better align with image embeddings
            embedding = text_embedding

        # Normalize embedding using the same function as in persistence.py
        embedding = normalize_embedding(embedding)
        print(f"Normalized embedding shape: {embedding.shape}")

        # Search for similar images
        results = db.find_similar_images(embedding)
        
        # Print results
        if not results:
            print("No similar images found.")
            return

        for result in results:
            print(f"Path: {result['file_path']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Captions: {result['captions']}")
            print("-" * 20)

if __name__ == "__main__":
    main()