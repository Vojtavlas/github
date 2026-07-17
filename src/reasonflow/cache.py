"""Reasoning-Selective Block Cache Manager (RSBCM)."""

import math
from dataclasses import dataclass
from typing import Dict


class RSBCMManager:
    """Evicts KV blocks by a score/depth priority for deep tree searches."""

    @dataclass
    class Block:
        block_id: int
        tree_depth: int
        branch_id: int
        importance: float

        @property
        def priority(self) -> float:
            return self.importance / (self.tree_depth + 1)

    def __init__(self, cfg):
        if cfg.max_blocks < 0:
            raise ValueError(f"max_blocks must be non-negative, got {cfg.max_blocks}")
        self.cfg = cfg
        self._pool: Dict[int, RSBCMManager.Block] = {}
        # _next_id is monotonically increasing per manager instance; overflow is
        # not a practical concern for normal use.
        self._next_id = 0
        self.eviction_events = 0

    def allocate(self, tree_depth: int, branch_id: int, importance: float = 1.0) -> int:
        if tree_depth < 0:
            raise ValueError(f"tree_depth must be non-negative, got {tree_depth}")
        if not math.isfinite(importance) or importance < 0:
            raise ValueError(
                f"importance must be a finite non-negative number, got {importance}"
            )

        bid = self._next_id
        self._next_id += 1
        new_block = self.Block(bid, tree_depth, branch_id, importance)

        # Decide which blocks to keep *before* storing, so the pool never
        # temporarily exceeds max_blocks.
        candidates = list(self._pool.values()) + [new_block]
        ordered = sorted(candidates, key=lambda b: (b.priority, b.block_id), reverse=True)
        keep = ordered[: self.cfg.max_blocks]
        self._pool = {b.block_id: b for b in keep}
        self.eviction_events += len(candidates) - len(keep)
        return bid

    def reset(self):
        self._pool = {}
        self._next_id = 0
        self.eviction_events = 0
