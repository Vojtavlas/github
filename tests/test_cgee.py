import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

# ``reasonflow.model_adapter`` does not exist in this worktree. Provide a
# runtime shim so the cgee module can be imported and unit-tested.
if "reasonflow.model_adapter" not in sys.modules:
    model_adapter_stub = types.ModuleType("reasonflow.model_adapter")
    model_adapter_stub.get_transformer_layers = lambda model: []
    sys.modules["reasonflow.model_adapter"] = model_adapter_stub

import reasonflow.cgee as cgee_module
from reasonflow.cgee import (
    CGEEAnalyzer,
    EarlyExitSignal,
    EarlyExitStrategy,
    EntropyTracker,
    ExitSignal,
    HookAdapter,
)
from reasonflow.config import RKSCConfig


class FakeLayer:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self._hook = None

    def register_forward_hook(self, fn):
        self._hook = fn
        return MagicMock()

    def __call__(self, x):
        if self._hook is not None:
            self._hook(self, (x,), self.output)
        return self.output


class FakeModel:
    def __init__(self, layers):
        self.model = SimpleNamespace(layers=layers)

    def __call__(self, *, input_ids, attention_mask=None):
        x = input_ids.float()
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=x)


def test_entropy_tracker_per_sample_entropy():
    tracker = EntropyTracker(torch.randn(100, 16))
    logits = torch.randn(2, 100)
    ent = tracker._entropy(logits)
    assert ent.shape == (2,)
    assert torch.all(ent >= 0)


def test_entropy_tracker_update():
    unembed = torch.randn(50, 16)
    tracker = EntropyTracker(unembed)
    hidden_state = torch.randn(2, 5, 16)

    entropy = tracker.update(0, hidden_state)

    assert isinstance(entropy, float)
    assert len(tracker.curve) == 1
    assert tracker.curve[0] == entropy
    assert entropy >= 0


def test_entropy_tracker_accepts_tuple_output():
    unembed = torch.randn(10, 8)
    tracker = EntropyTracker(unembed)
    h = torch.randn(1, 3, 8)

    ent1 = tracker.update(0, h)
    ent2 = tracker.update(1, (h,))

    assert len(tracker.curve) == 2
    assert ent1 == ent2


def test_early_exit_strategy_exits_when_entropy_low_and_stable():
    cfg = RKSCConfig(min_exit_layer=2, theta=1.0, entropy_stability_eps=0.5)
    strategy = EarlyExitStrategy(cfg)

    assert strategy.should_exit(2, 0.4, [0.5, 0.4]) is True


def test_early_exit_strategy_does_not_exit_prematurely():
    cfg = RKSCConfig(min_exit_layer=2, theta=1.0, entropy_stability_eps=0.5)
    strategy = EarlyExitStrategy(cfg)

    # Below minimum layer.
    assert strategy.should_exit(1, 0.4, [0.5, 0.4]) is False
    # Entropy too high.
    assert strategy.should_exit(2, 1.5, [1.6, 1.5]) is False
    # Not stable enough.
    assert strategy.should_exit(2, 0.4, [2.0, 0.4]) is False
    # Curve too short.
    assert strategy.should_exit(2, 0.4, [0.4]) is False


def test_hook_adapter_registers_and_invokes_callback():
    layers = [MagicMock(), MagicMock()]
    cgee_module.get_transformer_layers = lambda model: layers

    calls = []

    def callback(layer_idx, hidden_state):
        calls.append((layer_idx, tuple(hidden_state.shape)))
        return None

    adapter = HookAdapter(callback)
    model = MagicMock()
    adapter.register_hooks(model)

    assert len(adapter._handles) == 2
    layers[0].register_forward_hook.assert_called_once()
    layers[1].register_forward_hook.assert_called_once()

    hook_fn = layers[0].register_forward_hook.call_args[0][0]
    fake_output = torch.randn(1, 4, 16)
    hook_fn(layers[0], (torch.randn(1, 4, 16),), fake_output)

    assert calls == [(0, (1, 4, 16))]


def test_hook_adapter_stops_on_exit_signal():
    handles = [MagicMock(), MagicMock()]
    layers = [MagicMock(), MagicMock()]
    for layer, handle in zip(layers, handles):
        layer.register_forward_hook.return_value = handle
    cgee_module.get_transformer_layers = lambda model: layers

    callback_calls = []

    def callback(layer_idx, hidden_state):
        callback_calls.append(layer_idx)
        if layer_idx == 0:
            return ExitSignal(0)
        return None

    adapter = HookAdapter(callback)
    adapter.register_hooks(MagicMock())

    hook0 = layers[0].register_forward_hook.call_args[0][0]
    hook1 = layers[1].register_forward_hook.call_args[0][0]
    fake_output = torch.randn(1, 2, 16)

    with pytest.raises(EarlyExitSignal):
        hook0(layers[0], (fake_output,), fake_output)

    assert adapter.exit_signal.layer_idx == 0
    assert callback_calls == [0]

    # The second hook should be a no-op now.
    hook1(layers[1], (fake_output,), fake_output)
    assert callback_calls == [0]
    assert adapter.exit_signal.layer_idx == 0

    for handle in handles:
        handle.remove.assert_called()


def test_hook_adapter_remove_hooks():
    layers = [MagicMock(), MagicMock()]
    cgee_module.get_transformer_layers = lambda model: layers

    adapter = HookAdapter(lambda *args: None)
    adapter.register_hooks(MagicMock())
    adapter.remove_hooks()

    assert len(adapter._handles) == 0
    for layer in layers:
        layer.register_forward_hook.return_value.remove.assert_called_once()


def test_cgee_analyze_callback_exits():
    cfg = RKSCConfig(min_exit_layer=0, theta=100.0, entropy_stability_eps=1000.0)
    unembed = torch.randn(16, 16)
    analyzer = CGEEAnalyzer(cfg, unembed, 2)
    hidden_state = torch.randn(1, 3, 16)

    first = analyzer._callback(0, hidden_state)
    assert first is None
    assert len(analyzer.entropy_tracker.curve) == 1

    second = analyzer._callback(1, hidden_state)
    assert isinstance(second, ExitSignal)
    assert second.layer_idx == 1
    assert analyzer.exit_layer == 1
    assert analyzer._last_hidden is not None


def test_cgee_analyze_without_early_exit():
    cgee_module.get_transformer_layers = lambda model: []
    cfg = RKSCConfig()
    unembed = torch.randn(20, 16)
    analyzer = CGEEAnalyzer(cfg, unembed, 2)

    fake_logits = torch.randn(1, 20)
    model = MagicMock()
    model.return_value.logits = fake_logits.unsqueeze(1)

    logits, exit_layer, curve = analyzer.analyze(
        model, torch.tensor([[1]]), torch.tensor([[1]])
    )

    assert exit_layer is None
    assert curve == []
    assert torch.equal(logits, fake_logits)


def test_cgee_analyze_with_early_exit():
    cgee_module.get_transformer_layers = lambda model: model.model.layers
    cfg = RKSCConfig(min_exit_layer=0, theta=100.0, entropy_stability_eps=1000.0)
    unembed = torch.eye(16, 16)
    analyzer = CGEEAnalyzer(cfg, unembed, 2)

    hidden = torch.randn(1, 2, 16)
    layer0 = FakeLayer(hidden)
    layer1 = FakeLayer(hidden)
    model = FakeModel([layer0, layer1])

    logits, exit_layer, curve = analyzer.analyze(
        model, torch.tensor([[0]]), torch.tensor([[1]])
    )

    assert exit_layer == 1
    assert len(curve) == 2
    expected_logits = F.linear(hidden[:, -1, :].float(), unembed)
    assert torch.allclose(logits, expected_logits)


def test_skip_verification_relative_gap():
    cfg = RKSCConfig(gen_conf_threshold=0.70, relative_gap_threshold=0.10)
    analyzer = CGEEAnalyzer(cfg, torch.randn(10, 16), 2)
    assert analyzer.should_skip_verification([0.9, 0.7]) is True
    assert analyzer.should_skip_verification([0.9, 0.85]) is False
    assert analyzer.should_skip_verification([0.5, 0.3]) is False
