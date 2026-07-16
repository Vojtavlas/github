"""Reasoning-Selective Block Cache Manager (RSBCM)."""

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
        self.cfg = cfg
        self._pool: Dict[int, RSBCMManager.Block] = {}
        self._next_id = 0
        self.eviction_events = 0

    def allocate(self, tree_depth: int, branch_id: int, importance: float = 1.0) -> int:
        bid = self._next_id
        self._next_id += 1
        self._pool[bid] = self.Block(bid, tree_depth, branch_id, importance)
        self._maybe_evict()
        return bid

    def _maybe_evict(self):
        if len(self._pool) <= self.cfg.max_blocks:
            return
        n_evict = len(self._pool) - self.cfg.max_blocks
        ordered = sorted(self._pool.values(), key=lambda b: b.priority)
        for blk in ordered[-n_evict:]:
            del self._pool[blk.block_id]
            self.eviction_events += 1

    def reset(self):
        self._pool = {}
        self._next_id = 0
        self.eviction_events = 0
