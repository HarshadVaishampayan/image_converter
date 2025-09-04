import argparse

from sentence_transformers import SentenceTransformer
import caption_image
import persistence
from PIL import Image

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
    blip_name, clip_name, batch_size, beam_size = caption_image.load_models(env)

    clip_model = SentenceTransformer(clip_name)
    device = caption_image.get_device()

    if args.image:
        try:
            img = Image.open(args.image).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {args.image}")
            return
        except Exception as e:
            print(f"Error: Could not open image file: {e}")
            return

        # Create embeddings
        batch_embeddings = caption_image.create_embeddings(clip_model, device, [img])
        embedding = batch_embeddings[0].cpu().numpy()

        # Placeholder for persistence logic (replace with your actual code)
        print(f"Embedding shape: {embedding.shape}")
        # Example: Save embedding to a file
        # np.save("image_embedding.npy", embedding)

    elif args.query:
        # Placeholder for query processing logic (replace with your actual code)
        print(f"Processing query: {args.query}")
        # Example: Encode query string using CLIP
        query_embedding = clip_model.encode(args.query, convert_to_tensor=True, device=device)
        print(f"Query embedding shape: {query_embedding.shape}")

    db = persistence.PersistenceAPI()
    results = db.find_similar_images(embedding)

    for result in results:
        print(f"Path: {result['file_path']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Captions: {result['captions']}")
        print("-" * 20)

if __name__ == "__main__":
    main()