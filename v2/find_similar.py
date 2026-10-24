import argparse
import numpy as np
import os
import json
import re
import threading
from typing import List, Dict

from PIL import Image
from persistence import PersistenceAPI
from persistence import normalize_embedding
from model_manager import ModelManager
import tagging

def calculate_hybrid_score(query: str, captions: List[str], description: str, visual_distance: float) -> float:
    """
    Calculates a hybrid score by applying a keyword-based boost to the visual distance.
    Distance ranges from 0.0 (perfect) to 2.0.
    Boost can reduce the distance by up to 0.5.
    """
    if not query:
        return visual_distance
        
    # 1. Extract keywords from query (lowercase, alphanumeric)
    keywords = set(re.findall(r'\w+', query.lower()))
    if not keywords:
        return visual_distance
        
    # 2. Combine all available text for the image
    # We use whole-word matching to avoid partial matches (e.g., 'rose' vs 'rose-colored')
    full_text = (" ".join(captions) + " " + (description or "")).lower()
    
    # 3. Count how many query keywords appear as whole words in the text
    matches = 0
    for word in keywords:
        # \b ensures we match 'rose' but not 'rosemary' or 'rose-colored'
        if re.search(rf'\b{word}\b', full_text):
            matches += 1
            
    # 4. Calculate boost ratio (0.0 to 1.0)
    match_ratio = matches / len(keywords)
    
    # 5. Apply boost: subtract up to 0.5 from the distance
    # A perfect text match makes the image significantly more relevant
    boost = match_ratio * 0.5
    
    # Final score (lower is better)
    return max(0.0, visual_distance - boost)

# ---------------------------------------------------------------- text search ----
# Text queries are ranked by tagging.relevance(): SigLIP's calibrated image-text
# score, zero-shot tags rolled up the taxonomy, and description words, all 0..1.
# Image embeddings + metadata are cached per collection and refreshed when the
# collection changes (or when invalidate_cache() is called after indexing).
_cache = {}
_cache_lock = threading.Lock()


def invalidate_cache():
    with _cache_lock:
        _cache.clear()


def _image_table(db, model):
    """(ids, embeddings[N,D], metadatas, tags) for every image in the collection."""
    key = (os.path.realpath(db.path), db.collection_name)
    count = db.collection.count()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit["count"] == count:
            return hit["table"]
    r = db.collection.get(where={"type": "image"}, include=["embeddings", "metadatas"])
    metas = r["metadatas"] or []
    emb = np.asarray(r["embeddings"] if r["embeddings"] is not None else np.zeros((0, 1)), dtype=np.float32)
    if len(emb):
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    tags = [json.loads(m["tags"]) if m.get("tags") else None for m in metas]
    missing = [i for i, t in enumerate(tags) if t is None]
    if missing:  # images indexed before tagging existed: tag them now (in memory)
        for i, t in zip(missing, tagging.tag_embeddings(model, emb[missing])):
            tags[i] = t
    table = (r["ids"], emb, metas, tags)
    with _cache_lock:
        _cache[key] = {"count": count, "table": table}
    return table


def search_text(db, model, query_text: str, top_k: int = 6, sensitivity: float = 0.5) -> List[Dict]:
    """Rank all images for a text query. Lower hybrid_score is better (1 - relevance)."""
    ids, emb, metas, tags = _image_table(db, model)
    if not ids:
        return []
    scale, bias = tagging.logit_params(model)
    q = model.encode([f"a photo of {query_text.strip()}"], convert_to_tensor=True).float().cpu().numpy()[0]
    q /= max(np.linalg.norm(q), 1e-12)
    logits = scale * (emb @ q) + bias
    descriptions = [m.get("description", "") for m in metas]
    rel = tagging.relevance(query_text, logits, tags, descriptions)
    visual = tagging.confidence(logits)

    order = np.argsort(-np.asarray(rel))
    results = []
    for i in order:
        distance = 1.0 - rel[i]
        if distance > sensitivity or len(results) >= top_k:
            break
        m = metas[i]
        results.append({
            "image_id": m.get("image_id"),
            "file_path": m.get("file_path"),
            "file_name": m.get("file_name"),
            "description": m.get("description", ""),
            "captions": json.loads(m.get("captions") or "[]"),
            "tags": [label for label, _ in tagging.confident_tags(tags[i])],
            "hybrid_score": float(distance),
            "score": float(1.0 - visual[i]),
        })
    print(f"[{query_text!r}] {len(results)} result(s) within distance {sensitivity}")
    return results


def search_similar(query_text: str = None, image_path: str = None, force_rebuild: bool = False, top_k: int = 6, sensitivity: float = 0.5, embedding_model_type: str = "siglip") -> List[Dict]:
    """
    Search by text (tags + SigLIP relevance, see search_text) or by example image
    (nearest image embeddings with keyword re-ranking).
    """
    db = PersistenceAPI(embedding_model_type=embedding_model_type)
    if force_rebuild:
        print(f"Forcing ChromaDB collection rebuild for {embedding_model_type.upper()}...")
        db._rebuild_index()

    model_manager = ModelManager()
    model = model_manager.get_embedding_model(embedding_model_type)

    if query_text and not image_path and tagging.supports(model):
        return search_text(db, model, query_text, top_k=top_k, sensitivity=sensitivity)

    # 1. Generate Query Embedding
    if image_path:
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert('RGB')
        embedding = model.encode(img, convert_to_tensor=True)
    elif query_text:
        embedding = model.encode([query_text], convert_to_tensor=True)
    else:
        raise ValueError("Must provide either query_text or image_path")

    # Normalize embedding for consistent cosine distance
    embedding_np = embedding.cpu().numpy()
    # Handle single vector vs batch from encode
    if len(embedding_np.shape) > 1:
        embedding_np = embedding_np[0]
    embedding_list = normalize_embedding(embedding_np).tolist()
    
    # 2. Fetch Candidates (Fetch more than top_k for re-ranking)
    # Fetch 50 candidates to allow keyword matches to pull relevant but visually 'distant' images to the top
    candidate_results = db.find_similar_images(embedding_list, top_k=50)
    
    # 3. Apply Re-ranking (Hybrid Scoring)
    hybrid_results = []
    print("\n--- Semantic Search Raw Results (before Hybrid Re-ranking) ---")
    for i, res_minimal in enumerate(candidate_results):
        # Fetch full image data to get comprehensive descriptions and captions
        full_image_data = db.get_full_image_data(res_minimal['file_path'])
        
        if not full_image_data:
            print(f"  [WARNING] Could not retrieve full data for {res_minimal['file_path']}. Skipping.")
            continue

        # Aggregate all text for hybrid scoring
        all_text_content = []
        if full_image_data.get('description'):
            all_text_content.append(full_image_data['description'])
        if full_image_data.get('captions'):
            all_text_content.extend(full_image_data['captions'])
        
        # Log for debugging
        print(f"  {i+1}. File: {full_image_data['file_path']}")
        print(f"     Raw Visual Distance: {res_minimal['score']:.4f}")
        print(f"     Description: {full_image_data['description'][:100]}...") # Show full description
        print("-" * 20)
        
        visual_dist = res_minimal['score'] # ChromaDB returns cosine distance
        
        # Only re-rank if we have a text query
        if query_text and not image_path:
            final_score = calculate_hybrid_score(
                query=query_text,
                captions=full_image_data['captions'], # Use aggregated captions
                description=full_image_data['description'], # Use aggregated description
                visual_distance=visual_dist
            )
        else:
            final_score = visual_dist
            
        full_image_data['hybrid_score'] = final_score
        full_image_data['score'] = visual_dist # Keep original visual distance
        hybrid_results.append(full_image_data) # Append the full data
        
    # 4. Sort by Hybrid Score (lower is better)
    sorted_results = sorted(hybrid_results, key=lambda x: x['hybrid_score'])
    
    # 5. Filter by Sensitivity (Threshold)
    # With hybrid scoring, 0.5 is a very safe and solid threshold
    filtered_results = [r for r in sorted_results if r['hybrid_score'] <= sensitivity]
    
    print(f"[{embedding_model_type.upper()}] Fetched 50 candidates, returning {len(filtered_results[:top_k])} after Hybrid Re-ranking (threshold {sensitivity})")
    
    return filtered_results[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on image or query string.")
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--model", default="siglip", choices=['clip', 'siglip'], help="Model type to use.")
    parser.add_argument("--top-k", type=int, default=6, help="Top matches to return.")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Score threshold (lower is better).")
    args = parser.parse_args()

    try:
        results = search_similar(
            query_text=args.query,
            image_path=args.image,
            top_k=args.top_k,
            sensitivity=args.sensitivity,
            embedding_model_type=args.model
        )
            
        if not results:
            print("No similar images found.")
            return

        for result in results:
            print(f"Path: {result['file_path']}")
            print(f"Hybrid Score: {result['hybrid_score']:.4f} (Visual: {result['score']:.4f})")
            print(f"Captions: {result['captions']}") 
            print(f"Description: {result.get('description', '')[:100]}...")
            print("-" * 20)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
