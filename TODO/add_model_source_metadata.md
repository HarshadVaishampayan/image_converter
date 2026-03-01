# Add Model Source to Database Metadata

## Motivation
With the ability to choose different models for embeddings and captions (CLIP, SigLIP, Florence-2, Ollama), it is crucial to track *provenance*—which model generated which record. This allows for better debugging, filtering, and potentially comparing results from different models within the same database later.

## Criticality
**Medium**. It doesn't break functionality if missing, but it is essential for long-term data management and quality evaluation.

## Approach
1.  **PersistenceAPI:** Update the `insert_or_update` method in `v2/persistence.py`.
2.  **Metadata Injection:** When constructing the `base_metadata` dictionary:
    *   Add `embedding_model`: (e.g., "siglip-so400m")
    *   Add `caption_model`: (e.g., "florence-2-large" or "ollama-gemma3")
3.  **No Schema Change:** Since ChromaDB uses a JSON-like metadata store, no "ALTER TABLE" command is needed. We just start sending the new keys.

## Decision Log
*   *Decision:* Should we create separate collections for different caption models?
    *   *Reason:* No. The embedding model (`siglip` vs `clip`) dictates the vector space, so it needs separate collections (`images_siglip`, `images_clip`). The *caption* model just changes the *quality* of the text metadata associated with that vector. It can live in the same collection.

## Git Diff (Proposed)

```python
# v2/persistence.py

def insert_or_update(..., embedding_model_name, caption_model_name):
    # ...
    base_metadata = {
        # ... existing fields ...
        "embedding_model": embedding_model_name,
        "caption_model": caption_model_name,
    }
```
