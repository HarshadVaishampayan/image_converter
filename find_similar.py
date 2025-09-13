import argparse
import numpy as np

from sentence_transformers import SentenceTransformer
import caption_image
import persistence
from PIL import Image
from persistence import normalize_embedding  # Add this import

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on image or query string.")
    parser.add_argument("--image", type=str, help="Path to the image.")
    parser.add_argument("--query", type=str, help="Query string.")
    args = parser.parse_args()

    if not args.image and not args.query:
        print("Error: Please provide either an image path or a query string.")
        return

    # Detect environment and load models
    env = caption_image.detect_environment()
    print(f"Running in environment: {env}")
    
    # Load models
    blip_name, clip_name, batch_size, beam_size, device = caption_image.load_models(env)
    clip_model = SentenceTransformer(clip_name)
    db = persistence.PersistenceAPI()

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