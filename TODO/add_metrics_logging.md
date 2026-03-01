# Add Verbose Logging and Performance Metrics

## Objective
Enhance the observability of the ingestion pipeline. Users should see exactly what operation is currently happening (e.g., "Loading Image...", "Generating Description...", "Embedding...") and receive a performance summary at the end. This is crucial for identifying bottlenecks (e.g., is Ollama slow? Is disk I/O the problem?).

## Requirements
1.  **Metrics Class:** A singleton or context manager class to track:
    *   Start/End times for specific operations.
    *   Counts of operations.
    *   Calculation of averages (Total Time / Count).
2.  **Instrumentation:** Instrument `v2/caption_image.py` and `v2/persistence.py` to record:
    *   Image Loading time.
    *   Caption Generation time (Florence-2, currently skipped).
    *   Description Generation time (Ollama).
    *   Embedding Generation time (SigLIP/CLIP).
    *   Database Upsert time.
3.  **Verbose Logging:** Add `print` statements (or standard logging) to show progress *within* the batch loop, not just at the end.

## Design

### `v2/metrics.py` (New File)

```python
import time
from collections import defaultdict

class Metrics:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Metrics, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.current_starts = {}

    def start(self, name):
        self.current_starts[name] = time.time()

    def stop(self, name):
        if name in self.current_starts:
            duration = time.time() - self.current_starts[name]
            self.timings[name] += duration
            self.counts[name] += 1
            return duration
        return 0.0

    def print_summary(self):
        print("\n=== Performance Summary ===")
        for name in self.timings:
            count = self.counts[name]
            total = self.timings[name]
            avg = total / count if count > 0 else 0
            print(f"{name:<25}: {total:>6.2f}s total | {avg:>5.2f}s avg | {count} calls")
```

## Implementation Plan
1.  Create `v2/metrics.py`.
2.  Import `Metrics` in `v2/caption_image.py`.
3.  Add `metrics.start("Ollama Description")` and `metrics.stop(...)` around the `ThreadPoolExecutor` block.
4.  Add `metrics.start("Image Embedding")` around the SigLIP encode block.
5.  Add `metrics.start("DB Upsert")` in the persistence consumer.
6.  Print the summary at the end of the `process_folder_parallel_caption` generator.

