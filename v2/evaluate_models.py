import torch
from PIL import Image
from model_manager import ModelManager
import numpy as np
import os

def evaluate_models(image_path, search_terms):
    if not os.path.exists(image_path):
        print(f"Error: File not found {image_path}")
        return

    print(f"\n" + "="*80)
    print(f"EVALUATION: {image_path}")
    print("="*80)
    
    mm = ModelManager()
    device = mm.device
    
    # 1. Load Models
    print("Loading models (this might take a moment)...")
    clip = mm.get_embedding_model("clip")
    try:
        siglip = mm.get_embedding_model("siglip")
    except Exception as e:
        print(f"Error loading SigLIP: {e}")
        siglip = None
    
    img = Image.open(image_path).convert("RGB")
    
    # 2. Encode Image
    with torch.no_grad():
        clip_img_emb = clip.encode(img, convert_to_tensor=True)
        clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)
        
        if siglip:
            siglip_img_emb = siglip.encode(img, convert_to_tensor=True)
            siglip_img_emb = siglip_img_emb / siglip_img_emb.norm(dim=-1, keepdim=True)
    
    # 3. Compare Results
    print(f"\n{'Search Term':<25} | {'CLIP Sim \u2191':<12} | {'SigLIP Sim \u2191':<12} | {'Gap (Sig-CLIP)'}")
    print("-" * 80)
    
    results = []
    with torch.no_grad():
        for term in search_terms:
            # CLIP
            clip_text_emb = clip.encode([term], convert_to_tensor=True)
            # Ensure CLIP text embedding is normalized
            clip_text_emb = clip_text_emb / clip_text_emb.norm(dim=-1, keepdim=True)
            clip_sim = torch.dot(clip_img_emb.flatten(), clip_text_emb.flatten()).item()
            
            # SigLIP
            siglip_sim = 0.0
            if siglip:
                siglip_text_emb = siglip.encode([term], convert_to_tensor=True)
                # SigLIPWrapper already normalizes, but being explicit doesn't hurt
                siglip_text_emb = siglip_text_emb / siglip_text_emb.norm(dim=-1, keepdim=True)
                siglip_sim = torch.dot(siglip_img_emb.flatten(), siglip_text_emb.flatten()).item()
            
            gap = siglip_sim - clip_sim
            results.append((term, clip_sim, siglip_sim, gap))
    
    # Sort by SigLIP similarity
    for term, c_sim, s_sim, gap in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{term:<25} | {c_sim:<12.4f} | {s_sim:<12.4f} | {gap:+.4f}")

if __name__ == "__main__":
    # Test images from the user's previous examples
    images = [
        "/Users/aidev/Documents/pictures/DSC04967.JPG", # Rose
        "/Users/aidev/Documents/pictures/DSC08380.JPG"  # Cruise Ship
    ]
    
    terms = [
        "a red rose", 
        "a cruise ship", 
        "a green tree", 
        "water", 
        "petals", 
        "flowers", 
        "a boat in the ocean", 
        "nature", 
        "technology"
    ]
    
    for img_path in images:
        evaluate_models(img_path, terms)
