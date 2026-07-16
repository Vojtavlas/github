"""Confidence-Gated Early Exit (CGEE) for verification and decode."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import get_transformer_layers


class EarlyExitSignal(Exception):
    """Raised from a forward hook to stop the rest of the transformer layers."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx


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
        self._handles: list = []
        self.entropy_curve: List[float] = []
        self.exit_layer: Optional[int] = None
        self._last_hidden: Optional[torch.Tensor] = None

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * (probs + 1e-10).log()).sum(dim=-1)

    def _hook(self, layer_idx: int, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Use the last token of the sequence.
        last = h[:, -1, :].float()
        logits = F.linear(last, self.unembed.to(last.device))
        logits = logits.clamp(-1000.0, 1000.0)
        ent = self._entropy(logits)
        self.entropy_curve.append(ent.mean().item())
        n = len(self.entropy_curve)
        if (
            self.exit_layer is None
            and layer_idx >= self.cfg.min_exit_layer
            and n >= 2
            and ent.mean() < self.cfg.theta
            and abs(ent.mean().item() - self.entropy_curve[-2]) < self.cfg.entropy_stability_eps
        ):
            self.exit_layer = layer_idx
            self._last_hidden = last
            raise EarlyExitSignal(layer_idx)

    def register_hooks(self, model):
        layers = get_transformer_layers(model)
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._hook(idx, m, inp, out)
            )
            self._handles.append(handle)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def analyze(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int], List[float]]:
        """Run a forward pass, exiting early if entropy stabilises.

        Returns logits for the last token, the exit layer (None if no early exit),
        and the entropy curve.
        """
        self.entropy_curve = []
        self.exit_layer = None
        self._last_hidden = None
        self.register_hooks(model)
        out = None
        try:
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        except EarlyExitSignal:
            pass
        finally:
            self.remove_hooks()

        if self._last_hidden is not None:
            logits = F.linear(self._last_hidden, self.unembed.to(self._last_hidden.device))
        elif out is not None:
            logits = out.logits[:, -1, :]
        else:
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]

        return logits, self.exit_layer, self.entropy_curve

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
