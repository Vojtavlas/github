"""Tests for reasonflow.cache_adapter."""

import pytest
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
    select_kv_cache_rows,
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
    assert isinstance(expanded, tuple)
    assert expanded is not cache
    for (key, value), (orig_key, orig_value) in zip(expanded, cache):
        assert key.shape[0] == 3
        assert value.shape[0] == 3
        assert torch.equal(key[0], orig_key[0])
        assert torch.equal(value[0], orig_value[0])
        assert torch.equal(key[0], key[1]) and torch.equal(key[1], key[2])

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


def test_expand_invalid_batch_size():
    for batch_size in (0, -1):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            expand_kv(None, batch_size)

    tup = ((_make_tensors()[0], _make_tensors()[1]),)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        expand_kv(tup, 0)

    it_cache = IterableCache([_make_tensors()])
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        expand_kv(it_cache, -2)

    dc = DynamicCache()
    k, v = _make_tensors()
    dc.update(k, v, layer_idx=0)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        expand_kv(dc, 0)

    model_cache = FakeModelCache([_make_tensors()])
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        expand_kv(model_cache, -1)


def _make_batched_cache(num_layers: int = 2, batch: int = 3, seq_len: int = 4):
    """Build per-layer (key, value) tensors with distinct values per batch row."""
    pairs = []
    for layer in range(num_layers):
        key = torch.zeros(batch, 2, seq_len, 4)
        value = torch.zeros(batch, 2, seq_len, 4)
        for b in range(batch):
            key[b] = float(b + 1) + 0.1 * layer
            value[b] = float(b + 1) * 10.0 + 0.1 * layer
        pairs.append((key, value))
    return pairs


def _build_dynamic_cache(pairs):
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(pairs):
        cache.update(key, value, layer_idx=layer_idx)
    return cache


def _cache_rows(cache, indices):
    """Return list of (key, value) per layer for the requested batch rows."""
    rows = []
    for key, value in _cache_pairs(cache):
        rows.append((key[indices], value[indices]))
    return rows


def _assert_rows_equal(actual_cache, expected_rows):
    for (key, value), (exp_key, exp_value) in zip(_cache_pairs(actual_cache), expected_rows):
        assert torch.equal(key, exp_key)
        assert torch.equal(value, exp_value)


@pytest.mark.parametrize(
    "fixture_name",
    ["dynamic", "model_specific", "iterable", "tuple", "none"],
)
def test_select_kv_cache_rows(fixture_name):
    batch = 3
    indices = [0, 2]
    pairs = _make_batched_cache(num_layers=2, batch=batch, seq_len=4)

    if fixture_name == "dynamic":
        cache = _build_dynamic_cache(pairs)
    elif fixture_name == "model_specific":
        cache = FakeModelCache(pairs)
    elif fixture_name == "iterable":
        cache = IterableCache(pairs)
    elif fixture_name == "tuple":
        cache = tuple(pairs)
    elif fixture_name == "none":
        cache = None
    else:
        raise AssertionError(fixture_name)

    selected = select_kv_cache_rows(cache, indices)

    if fixture_name == "none":
        assert selected is None
        return

    for key, value in _cache_pairs(selected):
        assert key.shape[0] == len(indices)
        assert value.shape[0] == len(indices)

    expected_rows = _cache_rows(cache, indices)
    _assert_rows_equal(selected, expected_rows)


def test_select_kv_cache_rows_dynamic_preserves_type():
    pairs = _make_batched_cache(num_layers=2, batch=3, seq_len=4)
    cache = _build_dynamic_cache(pairs)
    selected = select_kv_cache_rows(cache, [1])
    assert isinstance(selected, DynamicCache)
    for key, value in _cache_pairs(selected):
        assert key.shape[0] == 1
        assert value.shape[0] == 1


def test_select_kv_cache_rows_tuple_preserves_type():
    pairs = _make_batched_cache(num_layers=2, batch=3, seq_len=4)
    cache = tuple(pairs)
    selected = select_kv_cache_rows(cache, [0, 1, 2])
    assert isinstance(selected, tuple)
    for (key, value), (orig_key, orig_value) in zip(selected, cache):
        assert torch.equal(key, orig_key)
        assert torch.equal(value, orig_value)


def test_select_kv_cache_rows_single_index():
    pairs = _make_batched_cache(num_layers=2, batch=3, seq_len=4)
    cache = FakeModelCache(pairs)
    selected = select_kv_cache_rows(cache, [2])
    assert isinstance(selected, FakeModelCache)
    for layer_idx, (key, value) in enumerate(zip(selected.key_cache, selected.value_cache)):
        assert key.shape[0] == 1
        assert torch.equal(key[0], cache.key_cache[layer_idx][2])
        assert torch.equal(value[0], cache.value_cache[layer_idx][2])


def test_select_kv_cache_rows_tensor_indices():
    pairs = _make_batched_cache(num_layers=2, batch=4, seq_len=3)
    cache = _build_dynamic_cache(pairs)
    idx = torch.tensor([3, 0])
    selected = select_kv_cache_rows(cache, idx)
    orig_pairs = list(_cache_pairs(cache))
    for (key, value), (orig_key, orig_value) in zip(_cache_pairs(selected), orig_pairs):
        assert key.shape[0] == 2
        assert torch.equal(key[0], orig_key[3])
        assert torch.equal(key[1], orig_key[0])

