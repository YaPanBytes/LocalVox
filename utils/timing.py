"""
Simple stage-timing helper so you can measure ASR/LLM/TTS latency
independently and confirm you're hitting the 2-second target.
"""

import time
from contextlib import contextmanager


@contextmanager
def timed_stage(name: str, log_fn=print):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log_fn(f"[timing] {name}: {elapsed*1000:.0f}ms")
