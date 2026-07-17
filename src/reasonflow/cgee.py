"""Confidence-Gated Early Exit (CGEE) for verification and decode."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model_adapter import get_transformer_layers


class ExitSignal:
    """Returned from a CGEE hook callback to indicate an early exit layer."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx


class EarlyExitSignal(Exception):
    """Raised by HookAdapter to stop the transformer forward pass early."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx


class EntropyTracker:
    """Compute and record the per-layer output entropy of a transformer."""

    def __init__(self, unembed_weight: torch.Tensor):
        self.unembed = unembed_weight.float()
        self.curve: List[float] = []

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Return the per-sample Shannon entropy of a logit distribution."""
        probs = torch.softmax(logits, dim=-1)
        return -(probs * (probs + 1e-10).log()).sum(dim=-1)

    def update(self, layer_idx: int, hidden_state: torch.Tensor) -> float:
        """Compute entropy for ``hidden_state`` and append it to the curve.

        ``hidden_state`` is expected to be a 3-D tensor of shape
        ``(batch, seq, hidden)``; the last token is used for the projection.
        A tuple ``(tensor, ...)`` is also accepted and unwrapped first.
        """
        if isinstance(hidden_state, tuple):
            h = hidden_state[0]
        else:
            h = hidden_state

        # Use the last token of the sequence.
        last = h[:, -1, :].float()
        logits = F.linear(last, self.unembed.to(last.device))
        logits = logits.clamp(-1000.0, 1000.0)
        ent = self._entropy(logits)
        self.curve.append(ent.mean().item())
        return self.curve[-1]


class EarlyExitStrategy:
    """Decide whether the transformer should exit early at a given layer."""

    def __init__(self, cfg):
        self.cfg = cfg

    def should_exit(self, layer_idx: int, current_entropy: float, curve: List[float]) -> bool:
        """Return True when the entropy is low and stable enough to exit.

        ``curve`` is the entropy history up to and including the current layer,
        i.e. ``curve[-1] == current_entropy``.
        """
        n = len(curve)
        return (
            layer_idx >= self.cfg.min_exit_layer
            and n >= 2
            and current_entropy < self.cfg.theta
            and abs(current_entropy - curve[-2]) < self.cfg.entropy_stability_eps
        )


class HookAdapter:
    """Register and remove PyTorch forward hooks that call a callback.

    The callback receives ``(layer_idx, hidden_state)`` and may return an
    :class:`ExitSignal` to stop further hooks from running.
    """

    def __init__(self, callback):
        self.callback = callback
        self._handles: list = []
        self.exit_signal: Optional[ExitSignal] = None

    def _hook(self, layer_idx: int, module, inputs, output):
        if self.exit_signal is not None:
            return

        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        signal = self.callback(layer_idx, h)
        if signal is not None:
            self.exit_signal = signal
            self.remove_hooks()
            raise EarlyExitSignal(signal.layer_idx)

    def register_hooks(self, model):
        """Register a forward hook on every transformer layer."""
        layers = get_transformer_layers(model)
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._hook(idx, m, inp, out)
            )
            self._handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class CGEEAnalyzer:
    """Two-level CGEE.

    * Level 1: skip the verification forward entirely when one branch is
      generation-confident and leads the others by a wide margin.
    * Level 2: when verification is necessary, exit early from the transformer
      stack once the per-layer output entropy becomes low and stable.
    """

    def __init__(self, cfg, unembed_weight: torch.Tensor, n_layers: int):
        self.cfg = cfg
        self.unembed = unembed_weight.float()
        self.n_layers = n_layers
        self.exit_strategy = EarlyExitStrategy(cfg)
        self.entropy_tracker = EntropyTracker(unembed_weight)
        self.exit_layer: Optional[int] = None
        self._last_hidden: Optional[torch.Tensor] = None

    def _callback(self, layer_idx: int, hidden_state: torch.Tensor):
        if isinstance(hidden_state, tuple):
            h = hidden_state[0]
        else:
            h = hidden_state

        last = h[:, -1, :].float()
        entropy = self.entropy_tracker.update(layer_idx, hidden_state)
        if self.exit_strategy.should_exit(layer_idx, entropy, self.entropy_tracker.curve):
            self.exit_layer = layer_idx
            self._last_hidden = last
            return ExitSignal(layer_idx)
        return None

    def analyze(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int], List[float]]:
        """Run a forward pass, exiting early if entropy stabilises.

        Returns logits for the last token, the exit layer (None if no early exit),
        and the entropy curve.
        """
        self.entropy_tracker = EntropyTracker(self.unembed)
        self.exit_layer = None
        self._last_hidden = None

        adapter = HookAdapter(self._callback)
        adapter.register_hooks(model)
        out = None
        try:
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        except EarlyExitSignal:
            pass
        finally:
            adapter.remove_hooks()

        if self._last_hidden is not None:
            logits = F.linear(self._last_hidden, self.unembed.to(self._last_hidden.device))
        elif out is not None:
            logits = out.logits[:, -1, :]
        else:
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]

        return logits, self.exit_layer, self.entropy_tracker.curve

    def should_skip_verification(self, confidences: List[float]) -> bool:
        """Level 1 gate: is the highest-confidence branch decisive?"""
        if not confidences:
            return False
        sorted_conf = sorted(confidences, reverse=True)
        top = sorted_conf[0]
        second = sorted_conf[1] if len(sorted_conf) > 1 else 0.0
        if top < self.cfg.gen_conf_threshold:
            return False
        if self.cfg.use_relative_gap:
            rel_gap = (top - second) / max(top, 1e-8)
            return rel_gap >= self.cfg.relative_gap_threshold
        return (top - second) >= self.cfg.gen_conf_gap
