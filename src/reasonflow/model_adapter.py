"""Model introspection adapters for locating transformer decoder layers."""

from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class ModelAdapter(ABC):
    """Abstract base class for architecture-specific layer extraction."""

    @abstractmethod
    def matches(self, model) -> bool:
        """Return True if this adapter can handle ``model``."""
        ...

    @abstractmethod
    def get_layers(self, model) -> List[nn.Module]:
        """Return the list of transformer decoder layers for ``model``."""
        ...


class LlamaAdapter(ModelAdapter):
    """Adapter for LLaMA-style models (``model.model.layers``)."""

    def matches(self, model) -> bool:
        return hasattr(model, "model") and hasattr(model.model, "layers")

    def get_layers(self, model) -> List[nn.Module]:
        return list(model.model.layers)


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2-style models (``model.transformer.h``)."""

    def matches(self, model) -> bool:
        return hasattr(model, "transformer") and hasattr(model.transformer, "h")

    def get_layers(self, model) -> List[nn.Module]:
        return list(model.transformer.h)


class GPTNeoXAdapter(ModelAdapter):
    """Adapter for GPT-NeoX-style models (``model.gpt_neox.layers``)."""

    def matches(self, model) -> bool:
        return hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers")

    def get_layers(self, model) -> List[nn.Module]:
        return list(model.gpt_neox.layers)


class HeuristicAdapter(ModelAdapter):
    """Fallback adapter that scans ``named_modules`` for a likely layer list."""

    def matches(self, model) -> bool:
        return True

    def get_layers(self, model) -> List[nn.Module]:
        for name, module in model.named_modules():
            if (
                name.count(".") == 2
                and ("layers" in name or name.endswith(".h"))
                and isinstance(module, nn.ModuleList)
            ):
                return list(module)
        raise ValueError("Could not locate transformer layers for CGEE hooks")


class ModelAdapterRegistry:
    """Registry that selects the first matching adapter for a model."""

    def __init__(self) -> None:
        self._adapters: List[ModelAdapter] = []

    def register(self, adapter: ModelAdapter) -> None:
        """Add ``adapter`` to the end of the search order."""
        self._adapters.append(adapter)

    def get_adapter(self, model) -> ModelAdapter:
        """Return the first adapter that matches ``model``."""
        for adapter in self._adapters:
            if adapter.matches(model):
                return adapter
        raise ValueError("No matching adapter found for model")

    def get_layers(self, model) -> List[nn.Module]:
        """Return transformer layers using the first matching adapter."""
        return self.get_adapter(model).get_layers(model)


# Module-level default registry.
_registry = ModelAdapterRegistry()
_registry.register(LlamaAdapter())
_registry.register(GPT2Adapter())
_registry.register(GPTNeoXAdapter())
_registry.register(HeuristicAdapter())


def get_transformer_layers(model) -> List[nn.Module]:
    """Locate the list of transformer decoder layers for ``model``."""
    return _registry.get_layers(model)
