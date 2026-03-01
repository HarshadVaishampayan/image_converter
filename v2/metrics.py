import time
from collections import defaultdict

class Metrics:
    """
    Singleton class to track performance metrics across the application.
    """
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
        """Start a timer for a specific operation."""
        self.current_starts[name] = time.time()

    def stop(self, name):
        """Stop the timer and record the duration."""
        if name in self.current_starts:
            duration = time.time() - self.current_starts[name]
            self.timings[name] += duration
            self.counts[name] += 1
            # Cleanup start time to handle re-entry correctly if needed, 
            # though simple start/stop usage implies serial per-key usage or different keys.
            del self.current_starts[name]
            return duration
        return 0.0

    def print_summary(self):
        """Print a formatted summary of all metrics."""
        print("\n" + "="*60)
        print(f"{ 'Operation':<30} | { 'Total':<10} | { 'Avg':<8} | { 'Count':<5}")
        print("-" * 60)
        for name in sorted(self.timings.keys()):
            count = self.counts[name]
            total = self.timings[name]
            avg = total / count if count > 0 else 0
            print(f"{name:<30} | {total:>7.2f}s | {avg:>6.2f}s | {count:>5}")
        print("="*60 + "\n")
