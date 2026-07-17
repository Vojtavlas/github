"""Tests for reasonflow.cache_adapter."""

import torch
from transformers import DynamicCache

from reasonflow.cache_adapter import (
    DynamicCacheAdapter,
    IterableCacheAdapter,
    LegacyTupleCacheAdapter,
    ModelSpecificCacheAdapter,
    clone_kv_cache,
    expand_kv,
    get_cache_adapter,
)


def _make_tensors(batch: int = 1, seq_len: int = 3, num_heads: int = 2, head_dim: int = 4):
    total = batch * num_heads * seq_len * head_dim
    key = torch.arange(total, dtype=torch.float32).reshape(batch, num_heads, seq_len, head_dim)
    value = key + 1000.0
    return key, value


def _cache_pairs(cache):
    """Yield (key, value) pairs from any supported cache format."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for key, value in zip(cache.key_cache, cache.value_cache):
            yield key, value
        return
    for item in cache:
        if isinstance(item, tuple):
            yield item[0], item[1]
        else:
            yield item


def _assert_cache_equal(a, b):
    for (ak, av), (bk, bv) in zip(_cache_pairs(a), _cache_pairs(b)):
        assert torch.equal(ak, bk)
        assert torch.equal(av, bv)


class FakeModelCache:
    """A model-specific cache with ``key_cache``/``value_cache`` lists."""

    def __init__(self, pairs):
        self.key_cache = [key for key, _ in pairs]
        self.value_cache = [value for _, value in pairs]


class IterableCache:
    """An older iterable cache object."""

    def __init__(self, pairs):
        self._pairs = pairs

    def update(self, key, value, layer_idx=0):
        """No-op; the adapter only needs the method for format detection."""

    def __iter__(self):
        for key, value in self._pairs:
            yield key, value


def test_none_cache():
    assert clone_kv_cache(None) is None
    assert expand_kv(None, 5) is None


def test_dynamic_cache_clone_and_expand():
    cache = DynamicCache()
    for i in range(2):
        key, value = _make_tensors(batch=1, seq_len=2 + i)
        cache.update(key, value, layer_idx=i)

    adapter = get_cache_adapter(cache)
    assert isinstance(adapter, DynamicCacheAdapter)

    cloned = clone_kv_cache(cache)
    assert cloned is not cache
    _assert_cache_equal(cache, cloned)

    # Clones are independent of the original.
    for key, _ in _cache_pairs(cache):
        key[0, 0, 0, 0] = 9999.0
    for (ckey, _), (okey, _) in zip(_cache_pairs(cloned), _cache_pairs(cache)):
        assert ckey[0, 0, 0, 0] != okey[0, 0, 0, 0]

    expanded = expand_kv(cache, 3)
    assert isinstance(expanded, DynamicCache)
    for key, value in _cache_pairs(expanded):
        assert key.shape[0] == 3
        assert value.shape[0] == 3
        assert torch.equal(key[0], key[1]) and torch.equal(key[1], key[2])

    assert expand_kv(cache, 1) is cache


def test_model_specific_cache_clone_and_expand():
    cache = FakeModelCache([_make_tensors(), _make_tensors(seq_len=5)])
    adapter = get_cache_adapter(cache)
    assert isinstance(adapter, ModelSpecificCacheAdapter)

    cloned = clone_kv_cache(cache)
    assert cloned is not cache
    assert isinstance(cloned, FakeModelCache)
    _assert_cache_equal(cache, cloned)
    assert cloned.key_cache is not cache.key_cache
    assert cloned.value_cache is not cache.value_cache
    assert cloned.key_cache[0] is not cache.key_cache[0]

    expanded = expand_kv(cache, 3)
    assert isinstance(expanded, FakeModelCache)
    assert expanded is not cache
    for key, value in zip(expanded.key_cache, expanded.value_cache):
        assert key.shape[0] == 3
        assert value.shape[0] == 3

    assert expand_kv(cache, 1) is cache


def test_model_specific_cache_uses_batch_repeat_interleave():
    class RepeatableModelCache(FakeModelCache):
        def __init__(self, pairs):
            super().__init__(pairs)
            self.repeat_called = False

        def batch_repeat_interleave(self, batch_size: int):
            self.repeat_called = True
            self.key_cache = [k.repeat_interleave(batch_size, dim=0) for k in self.key_cache]
            self.value_cache = [v.repeat_interleave(batch_size, dim=0) for v in self.value_cache]

    cache = RepeatableModelCache([_make_tensors(), _make_tensors(seq_len=5)])
    expanded = expand_kv(cache, 4)
    assert isinstance(expanded, RepeatableModelCache)
    assert expanded.repeat_called
    assert cache.repeat_called is False
    assert expanded.key_cache[0].shape[0] == 4


def test_iterable_cache_clone_and_expand():
    cache = IterableCache([_make_tensors(), _make_tensors(seq_len=5)])
    adapter = get_cache_adapter(cache)
    assert isinstance(adapter, IterableCacheAdapter)

    cloned = clone_kv_cache(cache)
    assert isinstance(cloned, DynamicCache)
    _assert_cache_equal(cache, cloned)

    expanded = expand_kv(cache, 3)
    assert isinstance(expanded, DynamicCache)
    for key, _ in _cache_pairs(expanded):
        assert key.shape[0] == 3

    assert expand_kv(cache, 1) is cache


def test_legacy_tuple_cache_clone_and_expand():
    key1, value1 = _make_tensors()
    key2, value2 = _make_tensors(seq_len=5)
    cache = ((key1, value1), (key2, value2))
    adapter = get_cache_adapter(cache)
    assert isinstance(adapter, LegacyTupleCacheAdapter)

    cloned = clone_kv_cache(cache)
    assert isinstance(cloned, tuple)
    assert cloned is not cache
    _assert_cache_equal(cache, cloned)

    expanded = expand_kv(cache, 3)
    assert isinstance(expanded, DynamicCache)
    for key, _ in _cache_pairs(expanded):
        assert key.shape[0] == 3

    assert expand_kv(cache, 1) is cache


def test_none_values_in_key_value_lists():
    cache = FakeModelCache([_make_tensors(), _make_tensors(seq_len=5)])
    cache.key_cache[1] = None
    cache.value_cache[0] = None

    cloned = clone_kv_cache(cache)
    assert cloned.key_cache[1] is None
    assert cloned.value_cache[0] is None
    assert cloned.key_cache[0] is not None
    assert cloned.value_cache[1] is not None
    assert cloned.key_cache[0] is not cache.key_cache[0]

    expanded = expand_kv(cache, 3)
    assert expanded.key_cache[1] is None
    assert expanded.value_cache[0] is None
    assert expanded.key_cache[0].shape[0] == 3
    assert expanded.value_cache[1].shape[0] == 3


def test_factory_prefers_dynamic_over_iterable():
    # DynamicCache also exposes update/__iter__, but should be routed to
    # DynamicCacheAdapter first.
    cache = DynamicCache()
    assert isinstance(get_cache_adapter(cache), DynamicCacheAdapter)
