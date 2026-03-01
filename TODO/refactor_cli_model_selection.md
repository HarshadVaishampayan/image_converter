# Refactor CLI and Model Selection

## Motivation
Currently, the selection of embedding models (CLIP vs. SigLIP) and captioning models (Florence-2 vs. Ollama) is partially hardcoded or coupled in a way that limits flexibility. The user wants to be able to explicitly choose which model generates captions and which model generates embeddings at runtime. Additionally, the system currently "hacks" the Florence-2 disabling by commenting out code, which is not sustainable.

## Criticality
**High**. This refactoring allows for proper experimentation and cleaner code architecture. It moves the project from a "hardcoded prototype" to a configurable tool.

## Approach
1.  **CLI Update:** Split the `--model` argument in `image_cli.py` into:
    *   `--embed-model`: Choices `['clip', 'siglip']` (Default: `siglip`)
    *   `--cap-model`: Choices `['florence2', 'ollama']` (Default: `florence2`)
2.  **Model Manager:** Restore the commented-out Florence-2 code in `v2/model_manager.py` but ensure it is only loaded when `cap_model='florence2'`.
3.  **Pipeline:** Refactor `v2/caption_image.py` to use a conditional block:
    *   If `cap_model == 'florence2'`: Run the Hugging Face inference loop.
    *   If `cap_model == 'ollama'`: Run the Ollama `describe_image` loop (using a simpler "describe" prompt for captioning).
    *   *Note:* The detailed description (Gemma 3) remains an Ollama task regardless of the caption model, as it provides the "deep" metadata.

## Decision Log
*   *Decision:* Keep `description` (Gemma 3) as a persistent secondary step?
    *   *Reason:* Yes, the detailed description is valuable for the hybrid search re-ranking. The `cap-model` choice specifically controls the *short* caption used for the primary text vector.

## Git Diff (Proposed)

```python
# v2/image_cli.py

@click.command()
@click.option('--embed-model', default='siglip', type=click.Choice(['clip', 'siglip']))
@click.option('--cap-model', default='florence2', type=click.Choice(['florence2', 'ollama']))
def describe(..., embed_model, cap_model):
    ...
```
