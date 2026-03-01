# Refactor Caption Pipeline for Modularity

## Motivation
The `process_folder_parallel_caption` function in `v2/caption_image.py` has become a monolith. It handles data loading, two different model inferences (PyTorch vs Ollama), memory management, and file writing. Adding the new "conditional model path" (Task 1) will make this function unreadable and error-prone.

## Criticality
**Medium**. Essential for maintainability before implementing Task 1.

## Approach
1.  **Extract Inference Logic:** Create helper functions:
    *   `_generate_captions_florence2(batch, model, processor, device)`
    *   `_generate_captions_ollama(batch_paths)`
    *   `_generate_embeddings(batch, model, device)`
2.  **Simplify Main Loop:** The main loop should look like:
    ```python
    for batch in loader:
        captions = _generate_captions(batch, model_choice)
        description = _generate_descriptions(batch)
        embeddings = _generate_embeddings(batch, embed_model)
        yield result
    ```

## Decision Log
*   *Decision:* Should we keep the `ThreadPoolExecutor` logic for Ollama?
    *   *Reason:* Yes, parallelizing the Ollama calls is the single biggest performance win we've achieved. That logic should be encapsulated in `_generate_descriptions`.

## Git Diff (Proposed)

```python
# v2/caption_image.py

def _generate_captions_florence2(...):
    # Moved Florence-2 generation logic here
    pass

def _generate_captions_ollama(...):
    # Moved Ollama description/caption logic here
    pass
```
