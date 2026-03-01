# Dynamic Hybrid Search Configuration

## Objective
Update the API (`find_similar.py`, `image_cli.py`) and the Web UI (`gui.py`, `grid.html`) to allow users to dynamically configure the matching algorithm. Users should be able to select which method is "Primary" and which is "Secondary," and control the "Influence" (boost strength) of the secondary method.

## Requirements
1.  **Parameters:**
    *   `primary`: `visual` (embeddings) | `text` (keyword matching).
    *   `secondary`: `visual` | `text` | `none`.
    *   `influence`: Float slider (`0.0` to `1.0`, default `0.15`).
2.  **Logic:**
    *   The `primary` method establishes the baseline ranking/score.
    *   The `secondary` method modifies that score based on the `influence` factor.
3.  **UI:** Add controls (dropdowns and slider) to the search form.

## Design & Algorithm

### Core Scoring Function (`calculate_hybrid_score`)

```python
def calculate_flexible_score(query, metadata, embedding_score, primary="visual", secondary="text", influence=0.15):
    # 1. Calculate Component Scores
    # Visual Score: Normalize from Distance (0=Good, 2=Bad) -> Similarity (1=Good, 0=Bad)
    # Using 1 - distance for rough similarity
    visual_score = 1.0 - embedding_score 
    
    # Text Score: 1.0 if keywords present, 0.0 if not
    text_score = calculate_text_match_ratio(query, metadata)
    
    # 2. Determine Baseline
    if primary == "visual":
        base_score = visual_score
        modifier_score = text_score if secondary == "text" else 0.0
    else: # primary == "text"
        base_score = text_score
        modifier_score = visual_score if secondary == "visual" else 0.0
        
    # 3. Apply Influence (Boost)
    # Formula: Final = Base + (Modifier * Influence)
    final_score = base_score + (modifier_score * influence)
    
    # 4. Invert back to "Distance" logic for sorting (Lower is Better)?
    # OR switch entire app to "Higher is Better" similarity logic.
    # DECISION: Stick to "Distance" (Lower is Better) for compatibility with existing ChromaDB logic.
    # Therefore:
    # Final Distance = BaseDist - (ModifierStrength * Influence)
    
    return final_distance
```

### Pitfalls & Risks
1.  **Metric Confusion:** ChromaDB returns *Distance* (Lower=Better). Keyword matching usually implies *Similarity* (Higher=Better). Mixing these requires careful inversion logic (`1 - dist`).
2.  **Primary=Text:** If Primary is "Text", we technically can't use ChromaDB's vector search as the retrieval mechanism. We would need a full-text search engine (BM25). 
    *   *Constraint:* For this iteration, `primary="text"` might effectively mean "Filter by text first, then sort by visual," OR we just interpret "Primary" as "The heavily weighted factor" but still perform retrieval via Vector (since we don't have a text index).
    *   *Decision:* Since we lack a dedicated Inverted Index (BM25), the *Retrieval* step must remain Vector-based (to find candidates). The *Ranking* step can prioritize text.
    *   *Risk:* If the vector search doesn't find the keyword-match image in the top 50, "Primary=Text" will fail to find it. This is a "Recall" limitation.

3.  **Influence Range:** If `influence` is too high (> 0.5), a keyword match might make a completely unrelated image score better than a perfect visual match (as seen previously).
    *   *Validation:* Limit slider to `0.0` - `0.5`.

## Code Changes

### 1. `v2/find_similar.py`
Update `search_similar` to accept the new params and implement the weighted logic.

```python
def search_similar(..., primary="visual", secondary="text", influence=0.15):
    # ... retrieval ...
    for res in candidate_results:
        # Calculate scores
        # ...
        if secondary == 'text':
             boost = match_ratio * influence
             final_score = visual_distance - boost
```

### 2. `v2/gui.py`
Update `image_grid` route to parse form data.

```python
primary = request.form.get('primary')
secondary = request.form.get('secondary')
influence = float(request.form.get('influence'))
```

### 3. `v2/templates/grid.html`
Add HTML controls.

```html
<select name="primary">...</select>
<select name="secondary">...</select>
<input type="range" name="influence" min="0" max="0.5" step="0.05">
```
