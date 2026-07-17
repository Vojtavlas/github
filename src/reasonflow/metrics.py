"""Simple metric helpers for benchmark scripts."""

from typing import Iterable, Tuple


def speedup(baseline_ms: float, optimized_ms: float) -> float:
    """Ratio-of-means speedup factor."""
    if baseline_ms <= 0:
        raise ValueError("baseline_ms must be > 0")
    if optimized_ms <= 0:
        raise ValueError("optimized_ms must be > 0")
    return baseline_ms / optimized_ms


def mean_speedup(measurements: Iterable[Tuple[float, float]]) -> float:
    """Given pairs of (baseline_ms, optimized_ms), return mean speedup."""
    bases, opts = [], []
    for base, opt in measurements:
        if base <= 0:
            raise ValueError("all baseline_ms values must be > 0")
        if opt <= 0:
            raise ValueError("all optimized_ms values must be > 0")
        bases.append(base)
        opts.append(opt)
    total_base = sum(bases)
    total_opt = sum(opts)
    if total_opt <= 0:
        raise ValueError("total optimized time must be > 0")
    return total_base / total_opt
