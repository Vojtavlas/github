"""Tests for the model introspection adapter registry."""

import pytest
import torch.nn as nn

from reasonflow.model_adapter import (
    GPT2Adapter,
    GPTNeoXAdapter,
    HeuristicAdapter,
    LlamaAdapter,
    ModelAdapterRegistry,
    get_transformer_layers,
)


def _make_layers(count: int = 3) -> nn.ModuleList:
    return nn.ModuleList([nn.Linear(8, 8) for _ in range(count)])


def _llama_model():
    class _LlamaLike:
        class _Inner:
            layers = _make_layers()

        model = _Inner()

    return _LlamaLike()


def _gpt2_model():
    class _GPT2Like:
        class _Inner:
            h = _make_layers()

        transformer = _Inner()

    return _GPT2Like()


def _gpt_neox_model():
    class _NeoXLike:
        class _Inner:
            layers = _make_layers()

        gpt_neox = _Inner()

    return _NeoXLike()


def _unknown_model():
    class _Unknown:
        layers = _make_layers()

        def named_modules(self):
            return iter([("a.b.layers", self.layers)])

    return _Unknown()


def _no_layers_model():
    class _Empty:
        def named_modules(self):
            return iter([])

    return _Empty()


def _non_modulelist_llama():
    class _Inner:
        layers = [1, 2, 3]

    class _Wrapper:
        model = _Inner()

    return _Wrapper()


def _non_modulelist_gpt2():
    class _Inner:
        h = [1, 2, 3]

    class _Wrapper:
        transformer = _Inner()

    return _Wrapper()


def _non_modulelist_gpt_neox():
    class _Inner:
        layers = [1, 2, 3]

    class _Wrapper:
        gpt_neox = _Inner()

    return _Wrapper()


def _empty_llama_model():
    class _Inner:
        layers = nn.ModuleList()

    class _Wrapper:
        model = _Inner()

    return _Wrapper()


def _empty_gpt2_model():
    class _Inner:
        h = nn.ModuleList()

    class _Wrapper:
        transformer = _Inner()

    return _Wrapper()


def _empty_gpt_neox_model():
    class _Inner:
        layers = nn.ModuleList()

    class _Wrapper:
        gpt_neox = _Inner()

    return _Wrapper()


def _heuristic_one_dot():
    class _Wrapper:
        layers = _make_layers()

        def named_modules(self):
            return iter([("model.layers", self.layers)])

    return _Wrapper()


def _heuristic_deep():
    class _Wrapper:
        layers = _make_layers()

        def named_modules(self):
            return iter([("a.b.c.layers", self.layers)])

    return _Wrapper()


def _heuristic_h():
    class _Wrapper:
        h = _make_layers()

        def named_modules(self):
            return iter([("transformer.h", self.h)])

    return _Wrapper()


def _heuristic_empty():
    class _Wrapper:
        layers = nn.ModuleList()

        def named_modules(self):
            return iter([("model.layers", self.layers)])

    return _Wrapper()


def test_llama_adapter():
    model = _llama_model()
    adapter = LlamaAdapter()
    assert adapter.matches(model)
    assert adapter.get_layers(model) == list(model.model.layers)


def test_gpt2_adapter():
    model = _gpt2_model()
    adapter = GPT2Adapter()
    assert adapter.matches(model)
    assert adapter.get_layers(model) == list(model.transformer.h)


def test_gpt_neox_adapter():
    model = _gpt_neox_model()
    adapter = GPTNeoXAdapter()
    assert adapter.matches(model)
    assert adapter.get_layers(model) == list(model.gpt_neox.layers)


def test_heuristic_adapter_finds_layers():
    model = _unknown_model()
    adapter = HeuristicAdapter()
    assert adapter.matches(model)
    assert adapter.get_layers(model) == list(model.layers)


def test_heuristic_adapter_raises_when_no_layers():
    model = _no_layers_model()
    adapter = HeuristicAdapter()
    assert adapter.matches(model)
    try:
        adapter.get_layers(model)
    except ValueError as exc:
        assert "Could not locate transformer layers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_registry_selects_first_match():
    registry = ModelAdapterRegistry()
    registry.register(LlamaAdapter())
    registry.register(HeuristicAdapter())

    assert isinstance(registry.get_adapter(_llama_model()), LlamaAdapter)
    assert isinstance(registry.get_adapter(_unknown_model()), HeuristicAdapter)


def test_registry_get_layers():
    registry = ModelAdapterRegistry()
    registry.register(LlamaAdapter())
    registry.register(HeuristicAdapter())

    llama = _llama_model()
    assert registry.get_layers(llama) == list(llama.model.layers)

    unknown = _unknown_model()
    assert registry.get_layers(unknown) == list(unknown.layers)


def test_get_transformer_layers_delegates_to_registry():
    llama = _llama_model()
    gpt2 = _gpt2_model()
    neox = _gpt_neox_model()
    unknown = _unknown_model()

    assert get_transformer_layers(llama) == list(llama.model.layers)
    assert get_transformer_layers(gpt2) == list(gpt2.transformer.h)
    assert get_transformer_layers(neox) == list(neox.gpt_neox.layers)
    assert get_transformer_layers(unknown) == list(unknown.layers)


def test_get_transformer_layers_raises_for_empty_model():
    try:
        get_transformer_layers(_no_layers_model())
    except ValueError as exc:
        assert "Could not locate transformer layers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_llama_adapter_rejects_non_modulelist():
    assert not LlamaAdapter().matches(_non_modulelist_llama())


def test_gpt2_adapter_rejects_non_modulelist():
    assert not GPT2Adapter().matches(_non_modulelist_gpt2())


def test_gpt_neox_adapter_rejects_non_modulelist():
    assert not GPTNeoXAdapter().matches(_non_modulelist_gpt_neox())


def test_heuristic_adapter_finds_one_dot_layers():
    model = _heuristic_one_dot()
    assert HeuristicAdapter().get_layers(model) == list(model.layers)


def test_heuristic_adapter_finds_deep_layers():
    model = _heuristic_deep()
    assert HeuristicAdapter().get_layers(model) == list(model.layers)


def test_heuristic_adapter_finds_h_layers():
    model = _heuristic_h()
    assert HeuristicAdapter().get_layers(model) == list(model.h)


def test_empty_modulelist_raises_valueerror():
    adapters = [
        ("llama", LlamaAdapter(), _empty_llama_model()),
        ("gpt2", GPT2Adapter(), _empty_gpt2_model()),
        ("gpt_neox", GPTNeoXAdapter(), _empty_gpt_neox_model()),
        ("heuristic", HeuristicAdapter(), _heuristic_empty()),
    ]
    for name, adapter, model in adapters:
        with pytest.raises(ValueError):
            adapter.get_layers(model)
