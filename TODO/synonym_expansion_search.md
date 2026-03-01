# Add Synonym Expansion for Hybrid Search

## Objective
Enhance the "Text Match" component of Hybrid Search by implementing **Query Expansion**. Instead of searching only for exact keywords (e.g., "Ocean"), the system should automatically generate synonyms (e.g., "Sea", "Water", "Lake") and match against any of them. This improves recall and handles vocabulary mismatches between the user and the image description.

## Requirements
1.  **Model Manager Update:**
    *   Add a new placeholder/loader for a "Synonym Model" (likely a small, fast LLM via Ollama, e.g., `llama3.2:1b` or reusing `gemma3`).
    *   This model must be separate from the Caption/Description models to allow for optimizing latency (a tiny model is best for simple synonym tasks).
2.  **API/CLI Update:**
    *   Add argument `--synonym-model` to `search` command.
    *   Add flag `--expand-query` (default: False) to enable this behavior.
3.  **GUI Update:**
    *   Add a checkbox "Expand Query (Synonyms)" to the search form.
    *   Pass this preference to the backend.
4.  **Logic Update (`find_similar.py`):**
    *   Before `calculate_hybrid_score`, call the Synonym Model.
    *   Prompt: *"List 5 synonyms for '{query}'. Return only the words, comma-separated."*
    *   Pass the *expanded* list of keywords to the regex matcher.

## Design

### 1. `v2/model_manager.py`
Add `get_synonym_model()` which essentially wraps an Ollama client for text-only generation.

```python
def get_synonym_model(self):
    # Returns a callable or client for the lightweight synonym LLM
    pass
```

### 2. `v2/find_similar.py`
Modify `search_similar` to handle expansion.

```python
def expand_query(query, model_name):
    # Call Ollama: "Synonyms for {query}?"
    # Return: "ocean, sea, water, waves"
    return expanded_keywords_list

def search_similar(..., expand_query=False, synonym_model="llama3.2:1b"):
    keywords = [query]
    if expand_query:
        synonyms = expand_query(query, synonym_model)
        keywords.extend(synonyms)
    
    # ... pass 'keywords' list to calculate_hybrid_score instead of just 'query' ...
```

## Risks & Mitigations
*   **Latency:** Calling an LLM during search adds time.
    *   *Mitigation:* Use the smallest possible model (`qwen:0.5b` or `llama3.2:1b`).
*   **Drift:** Synonyms might get too loose ("Ocean" -> "Blue" -> "Sad").
    *   *Mitigation:* Limit expansion to ~3-5 words and keep the prompt strict ("direct synonyms only").

