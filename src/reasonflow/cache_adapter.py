"""Adapter layer for cloning and expanding KV caches across formats."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import torch
from transformers import DynamicCache


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")


def _as_index_tensor(indices: Union[Sequence[int], torch.Tensor]) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        idx = indices.to(dtype=torch.long)
    else:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    return idx


class CacheAdapter(ABC):
    """Abstract base class for KV-cache cloning and expansion."""

    @abstractmethod
    def clone(self, cache: Any) -> Any:
        """Return a deep copy of ``cache`` that does not share tensors."""
        ...

    @abstractmethod
    def expand(self, cache: Any, batch_size: int) -> Any:
        """Return ``cache`` expanded from batch-1 to ``batch_size``."""
        ...

    @abstractmethod
    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        """Return ``cache`` with only the batch rows in ``indices``."""
        ...


class _NoneCacheAdapter(CacheAdapter):
    """Adapter for ``None`` caches."""

    def clone(self, cache: Any) -> Any:
        return None

    def expand(self, cache: Any, batch_size: int) -> Any:
        _validate_batch_size(batch_size)
        return None

    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        return None


class DynamicCacheAdapter(CacheAdapter):
    """Adapter for Hugging Face ``DynamicCache`` objects.

    Handles both the legacy ``key_cache``/``value_cache`` list layout and
    newer implementations that expose ``update``/``__iter__`` and a
    ``batch_repeat_interleave`` helper.
    """

    def clone(self, cache: Any) -> Any:
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            new_cache = copy.deepcopy(cache)
            new_cache.key_cache = [
                k.clone() if k is not None else None for k in cache.key_cache
            ]
            new_cache.value_cache = [
                v.clone() if v is not None else None for v in cache.value_cache
            ]
            return new_cache

        # Fallback for newer DynamicCache implementations backed by layers.
        new_cache = copy.deepcopy(cache)
        for layer in new_cache.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys = layer.keys.clone()
            if hasattr(layer, "values") and layer.values is not None:
                layer.values = layer.values.clone()
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        _validate_batch_size(batch_size)
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            expanded = copy.deepcopy(cache)
            expanded.key_cache = [
                k.repeat_interleave(batch_size, dim=0) if k is not None else None
                for k in cache.key_cache
            ]
            expanded.value_cache = [
                v.repeat_interleave(batch_size, dim=0) if v is not None else None
                for v in cache.value_cache
            ]
            return expanded

        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
                layer_idx=layer_idx,
            )
        return new_cache

    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        idx = _as_index_tensor(indices)

        if hasattr(cache, "batch_select_indices"):
            selected = self.clone(cache)
            selected.batch_select_indices(idx)
            return selected

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            selected = copy.deepcopy(cache)
            selected.key_cache = [
                k.index_select(0, idx) if k is not None else None for k in cache.key_cache
            ]
            selected.value_cache = [
                v.index_select(0, idx) if v is not None else None for v in cache.value_cache
            ]
            return selected

        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.index_select(0, idx),
                value_states.index_select(0, idx),
                layer_idx=layer_idx,
            )
        return new_cache


class ModelSpecificCacheAdapter(CacheAdapter):
    """Adapter for model-specific cache subclasses with ``key_cache``/``value_cache``."""

    def clone(self, cache: Any) -> Any:
        new_cache = copy.deepcopy(cache)
        new_cache.key_cache = [
            k.clone() if k is not None else None for k in cache.key_cache
        ]
        new_cache.value_cache = [
            v.clone() if v is not None else None for v in cache.value_cache
        ]
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        _validate_batch_size(batch_size)
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        expanded = copy.deepcopy(cache)
        expanded.key_cache = [
            k.repeat_interleave(batch_size, dim=0) if k is not None else None
            for k in cache.key_cache
        ]
        expanded.value_cache = [
            v.repeat_interleave(batch_size, dim=0) if v is not None else None
            for v in cache.value_cache
        ]
        return expanded

    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        idx = _as_index_tensor(indices)
        if hasattr(cache, "batch_select_indices"):
            selected = self.clone(cache)
            selected.batch_select_indices(idx)
            return selected
        selected = copy.deepcopy(cache)
        selected.key_cache = [
            k.index_select(0, idx) if k is not None else None for k in cache.key_cache
        ]
        selected.value_cache = [
            v.index_select(0, idx) if v is not None else None for v in cache.value_cache
        ]
        return selected


class IterableCacheAdapter(CacheAdapter):
    """Adapter for older iterable cache objects with ``update`` and ``__iter__``.

    Because arbitrary iterables may not support batch expansion, ``expand``
    returns a ``DynamicCache`` for ``batch_size > 1``.
    """

    def clone(self, cache: Any) -> Any:
        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(key_states.clone(), value_states.clone(), layer_idx=layer_idx)
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        _validate_batch_size(batch_size)
        if batch_size == 1:
            return cache

        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
                layer_idx=layer_idx,
            )
        return new_cache

    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        idx = _as_index_tensor(indices)
        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.index_select(0, idx),
                value_states.index_select(0, idx),
                layer_idx=layer_idx,
            )
        return new_cache


class LegacyTupleCacheAdapter(CacheAdapter):
    """Adapter for legacy tuple caches: ``tuple`` of ``(key, value)`` tensors."""

    def clone(self, cache: Any) -> Any:
        return tuple((k.clone(), v.clone()) for k, v in cache)

    def expand(self, cache: Any, batch_size: int) -> Any:
        _validate_batch_size(batch_size)
        if batch_size == 1:
            return cache

        return tuple(
            (
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
            )
            for key_states, value_states in cache
        )

    def select(self, cache: Any, indices: Union[Sequence[int], torch.Tensor]) -> Any:
        idx = _as_index_tensor(indices)
        return tuple(
            (key_states.index_select(0, idx), value_states.index_select(0, idx))
            for key_states, value_states in cache
        )


_none_adapter = _NoneCacheAdapter()


def get_cache_adapter(cache: Any) -> CacheAdapter:
    """Return the appropriate adapter for ``cache``.

    ``None`` is handled by a no-op adapter.  Selection order mirrors the
    precedence in the original ``utils.py`` helpers.
    """
    if cache is None:
        return _none_adapter
    if isinstance(cache, DynamicCache):
        return DynamicCacheAdapter()
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return ModelSpecificCacheAdapter()
    if hasattr(cache, "update") and hasattr(cache, "__iter__"):
        return IterableCacheAdapter()
    if isinstance(cache, tuple):
        return LegacyTupleCacheAdapter()
    if hasattr(cache, "__iter__"):
        return LegacyTupleCacheAdapter()
    raise TypeError(f"Unsupported KV cache type: {type(cache)}")


def clone_kv_cache(pkv: Any) -> Any:
    """Return a deep copy of a prefix KV cache."""
    return get_cache_adapter(pkv).clone(pkv)


def expand_kv(pkv: Any, batch_size: int) -> Any:
    """Expand a batch-1 KV cache to ``batch_size``."""
    return get_cache_adapter(pkv).expand(pkv, batch_size)


def select_kv_cache_rows(
    pkv: Any, indices: Union[Sequence[int], torch.Tensor]
) -> Any:
    """Return a KV cache containing only the batch rows in ``indices``.

    The returned cache is the same format as ``pkv`` (or ``DynamicCache`` for
    iterable caches that cannot be selected in place) and shares no storage
    with the original. ``None`` caches return ``None``.
    """
    return get_cache_adapter(pkv).select(pkv, indices)


__all__ = [
    "CacheAdapter",
    "DynamicCacheAdapter",
    "ModelSpecificCacheAdapter",
    "IterableCacheAdapter",
    "LegacyTupleCacheAdapter",
    "get_cache_adapter",
    "clone_kv_cache",
    "expand_kv",
    "select_kv_cache_rows",
]
