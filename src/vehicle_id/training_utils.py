"""Small helpers for choosing a batch size that fits the GPU."""
from __future__ import annotations
from typing import Callable

BATCH_CANDIDATES = (20, 16, 12, 10, 8, 6, 4)


def pick_batch_size(start: int, try_batch: Callable[[int], float | None], max_memory_fraction: float = 0.8) -> int:
    """Return the largest batch whose test step fits. try_batch gives peak memory use (0 to 1), or None if out of memory."""
    candidates = sorted({start, *(size for size in BATCH_CANDIDATES if size < start)}, reverse=True)
    for size in candidates:
        fraction = try_batch(size)
        status = "out of memory" if fraction is None else f"{fraction:.0%} of GPU memory"
        print(f"Batch size {size}: {status}", flush=True)
        if fraction is not None and fraction <= max_memory_fraction:
            return size
    raise SystemExit("No batch size fits this GPU. Set USE_TINY_MODEL = True in the notebook.")


def accumulation_for(batch_size: int, effective_batch: int) -> int:
    """Gradient accumulation steps that keep the effective batch size about the same."""
    return max(1, round(effective_batch / batch_size))
