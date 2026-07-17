"""Tests for the RSBCM cache manager eviction policy."""

import pytest

from reasonflow.cache import RSBCMManager
from reasonflow.config import RSBCMConfig


def test_no_eviction_under_capacity():
    cfg = RSBCMConfig(max_blocks=5)
    mgr = RSBCMManager(cfg)
    for i in range(5):
        mgr.allocate(tree_depth=i, branch_id=i, importance=1.0)
    assert len(mgr._pool) == 5
    assert mgr.eviction_events == 0


def test_evicts_lowest_priority_block():
    """A correct policy drops the LOWEST-priority block when over capacity.

    Priority is `importance / (tree_depth + 1)`, so a shallow, high-importance
    block must be retained over a deep, low-importance one.
    """
    cfg = RSBCMConfig(max_blocks=2)
    mgr = RSBCMManager(cfg)

    # block 0: high priority (importance=10, depth=0 -> priority=10.0)
    high = mgr.allocate(tree_depth=0, branch_id=0, importance=10.0)
    # block 1: low priority (importance=0.1, depth=5 -> priority~=0.0167)
    low = mgr.allocate(tree_depth=5, branch_id=1, importance=0.1)
    # block 2: medium priority (importance=1.0, depth=1 -> priority=0.5)
    medium = mgr.allocate(tree_depth=1, branch_id=2, importance=1.0)

    assert len(mgr._pool) == cfg.max_blocks
    assert mgr.eviction_events == 1

    # The low-priority block must be the one evicted; high and medium survive.
    assert high in mgr._pool
    assert medium in mgr._pool
    assert low not in mgr._pool


def test_reset_clears_pool():
    mgr = RSBCMManager(RSBCMConfig(max_blocks=10))
    mgr.allocate(tree_depth=0, branch_id=0, importance=1.0)
    mgr.allocate(tree_depth=1, branch_id=1, importance=1.0)
    assert len(mgr._pool) == 2
    mgr.reset()
    assert len(mgr._pool) == 0
    assert mgr.eviction_events == 0
    assert mgr._next_id == 0


def test_pool_never_exceeds_max_blocks_during_allocation():
    """The pool must never contain more than max_blocks blocks."""

    class Checked(RSBCMManager):
        def allocate(self, *args, **kwargs):
            bid = super().allocate(*args, **kwargs)
            assert len(self._pool) <= self.cfg.max_blocks
            return bid

    cfg = RSBCMConfig(max_blocks=3)
    mgr = Checked(cfg)
    for i in range(10):
        mgr.allocate(tree_depth=i, branch_id=i, importance=1.0)
    assert len(mgr._pool) == cfg.max_blocks


def test_negative_tree_depth_raises():
    mgr = RSBCMManager(RSBCMConfig(max_blocks=5))
    with pytest.raises(ValueError):
        mgr.allocate(tree_depth=-1, branch_id=0, importance=1.0)


def test_negative_importance_raises():
    mgr = RSBCMManager(RSBCMConfig(max_blocks=5))
    with pytest.raises(ValueError):
        mgr.allocate(tree_depth=0, branch_id=0, importance=-1.0)


def test_nonfinite_importance_raises():
    mgr = RSBCMManager(RSBCMConfig(max_blocks=5))
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            mgr.allocate(tree_depth=0, branch_id=0, importance=bad)


def test_max_blocks_zero_evicts_immediately():
    """With max_blocks=0 every newly allocated block is evicted immediately."""
    cfg = RSBCMConfig(max_blocks=0)
    mgr = RSBCMManager(cfg)
    bids = [mgr.allocate(tree_depth=0, branch_id=i, importance=1.0) for i in range(5)]
    assert len(mgr._pool) == 0
    assert mgr.eviction_events == 5
    assert all(isinstance(bid, int) for bid in bids)


def test_negative_max_blocks_raises():
    with pytest.raises(ValueError):
        RSBCMManager(RSBCMConfig(max_blocks=-1))
