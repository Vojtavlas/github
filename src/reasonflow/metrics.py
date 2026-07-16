"""Simple metric helpers for benchmark scripts."""

from typing import Iterable, Tuple


def speedup(baseline_ms: float, optimized_ms: float) -> float:
    """Ratio-of-means speedup factor."""
    if optimized_ms <= 0:
        return 1.0
    return baseline_ms / optimized_ms


def mean_speedup(measurements: Iterable[Tuple[float, float]]) -> float:
    """Given pairs of (baseline_ms, optimized_ms), return mean speedup."""
    bases, opts = [], []
    for base, opt in measurements:
        bases.append(base)
        opts.append(opt)
    total_base = sum(bases)
    total_opt = sum(opts)
    return total_base / max(total_opt, 1e-9)
