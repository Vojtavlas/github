"""Adapter layer for cloning and expanding KV caches across formats."""

import copy
from abc import ABC, abstractmethod
from typing import Any

from transformers import DynamicCache


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


class _NoneCacheAdapter(CacheAdapter):
    """Adapter for ``None`` caches."""

    def clone(self, cache: Any) -> Any:
        return None

    def expand(self, cache: Any, batch_size: int) -> Any:
        return None


class DynamicCacheAdapter(CacheAdapter):
    """Adapter for Hugging Face ``DynamicCache`` objects.

    Handles both the legacy ``key_cache``/``value_cache`` list layout and
    newer implementations that expose ``update``/``__iter__`` and a
    ``batch_repeat_interleave`` helper.
    """

    def clone(self, cache: Any) -> Any:
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            new_cache = copy.copy(cache)
            new_cache.key_cache = [
                k.clone() if k is not None else None for k in cache.key_cache
            ]
            new_cache.value_cache = [
                v.clone() if v is not None else None for v in cache.value_cache
            ]
            return new_cache

        # Fallback for newer DynamicCache implementations backed by layers.
        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(key_states.clone(), value_states.clone(), layer_idx=layer_idx)
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            expanded = copy.copy(cache)
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


class ModelSpecificCacheAdapter(CacheAdapter):
    """Adapter for model-specific cache subclasses with ``key_cache``/``value_cache``."""

    def clone(self, cache: Any) -> Any:
        new_cache = copy.copy(cache)
        new_cache.key_cache = [
            k.clone() if k is not None else None for k in cache.key_cache
        ]
        new_cache.value_cache = [
            v.clone() if v is not None else None for v in cache.value_cache
        ]
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        expanded = copy.copy(cache)
        expanded.key_cache = [
            k.repeat_interleave(batch_size, dim=0) if k is not None else None
            for k in cache.key_cache
        ]
        expanded.value_cache = [
            v.repeat_interleave(batch_size, dim=0) if v is not None else None
            for v in cache.value_cache
        ]
        return expanded


class IterableCacheAdapter(CacheAdapter):
    """Adapter for older iterable cache objects with ``update`` and ``__iter__``."""

    def clone(self, cache: Any) -> Any:
        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(key_states.clone(), value_states.clone(), layer_idx=layer_idx)
        return new_cache

    def expand(self, cache: Any, batch_size: int) -> Any:
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
                layer_idx=layer_idx,
            )
        return new_cache


class LegacyTupleCacheAdapter(CacheAdapter):
    """Adapter for legacy tuple caches: ``tuple`` of ``(key, value)`` tensors."""

    def clone(self, cache: Any) -> Any:
        return tuple((k.clone(), v.clone()) for k, v in cache)

    def expand(self, cache: Any, batch_size: int) -> Any:
        if batch_size == 1:
            return cache

        if hasattr(cache, "batch_repeat_interleave"):
            expanded = self.clone(cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(cache):
            new_cache.update(
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
                layer_idx=layer_idx,
            )
        return new_cache


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


__all__ = [
    "CacheAdapter",
    "DynamicCacheAdapter",
    "ModelSpecificCacheAdapter",
    "IterableCacheAdapter",
    "LegacyTupleCacheAdapter",
    "get_cache_adapter",
    "clone_kv_cache",
    "expand_kv",
]
