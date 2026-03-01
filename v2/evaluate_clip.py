import torch
from PIL import Image
from model_manager import ModelManager
import numpy as np

def evaluate_embeddings(image_path, search_terms):
    print(f"\n--- CLIP Evaluation: {image_path} ---")
    mm = ModelManager()
    device = mm.device
    clip = mm.get_clip_model()
    
    # 1. Load and Encode Image
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        img_emb = clip.encode(img, convert_to_tensor=True)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True) # Normalize
    
    # 2. Encode and Compare Text Terms
    results = []
    with torch.no_grad():
        for term in search_terms:
            text_emb = clip.encode([term], convert_to_tensor=True)[0]
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True) # Normalize
            
            # Cosine Similarity (Dot product of normalized vectors)
            similarity = torch.dot(img_emb, text_emb).item()
            # Convert to distance (same as ChromaDB cosine)
            distance = 1.0 - similarity
            results.append((term, similarity, distance))
    
    # 3. Print Results
    print(f"{'Search Term':<20} | {'Similarity \u2191':<12} | {'Distance \u2193 (Chroma)'}")
    print("-" * 60)
    for term, sim, dist in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{term:<20} | {sim:<12.4f} | {dist:.4f}")

if __name__ == "__main__":
    # Example usage:
    terms = ["a red rose", "a cruise ship", "a green tree", "water", "petals", "flowers"]
    # Change these paths to actual files on your Mac to test
    evaluate_embeddings("/Users/aidev/Documents/pictures/DSC04967.JPG", terms)
    evaluate_embeddings("/Users/aidev/Documents/pictures/DSC08380.JPG", terms)
