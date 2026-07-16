"""Attention-Similarity KV Sharing (ASKS)."""

from typing import Dict, List, Optional

import torch

from .utils import squeeze_hidden


class ASKSManager:
    """Gates prefix KV reuse by layer-wise hidden-state cosine similarity.

    ASKS strictly generalises token-exact prefix caching: a branch that is
    lexically identical to the root has similarity 1.0 and always reuses the
    cache; a branch that is only semantically close can also be accepted when
    its weighted cosine similarity exceeds ``tau``.
    """

    def __init__(self, cfg, arch: dict):
        self.cfg = cfg
        self.arch = arch
        self.root_hidden: Optional[torch.Tensor] = None
        self.root_kv: Optional[object] = None
        self.records: Dict[int, bool] = {}

    def capture_root(self, kv_cache, hidden_states: List[torch.Tensor]):
        """Store root prefix KV and layer-wise hidden states (unit-normalised)."""
        if hidden_states:
            vecs = [squeeze_hidden(h).detach() for h in hidden_states]
            stacked = torch.stack(vecs)  # [n_layers, hidden_dim]
            norms = stacked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.root_hidden = stacked / norms
        else:
            self.root_hidden = None
        self.root_kv = kv_cache

    def similarity(self, branch_hidden: List[torch.Tensor]) -> float:
        """Weighted cosine similarity between a branch and the root."""
        if self.root_hidden is None or not branch_hidden:
            return 0.0
        dev = self.root_hidden.device
        b = torch.stack([squeeze_hidden(h).to(dev) for h in branch_hidden])
        b_norms = b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_unit = b / b_norms
        cos = (self.root_hidden.to(dev) * b_unit).sum(-1)  # [n_layers]
        n = cos.shape[0]
        # Emphasise later layers, which carry more task-relevant semantics.
        weights = torch.exp(torch.linspace(0.0, 1.5, n, device=dev))
        weights = weights / weights.sum()
        return float((cos * weights).sum().item())

    def score_branch(self, branch_id: int, branch_hidden: List[torch.Tensor]) -> bool:
        """Return True iff the branch may reuse the root prefix KV cache."""
        sim = self.similarity(branch_hidden)
        reuse = sim >= self.cfg.tau
        self.records[branch_id] = reuse
        return reuse

    def get_root_kv(self, branch_id: int) -> Optional[object]:
        """Return cached KV for this branch if the gate allowed reuse."""
        if self.records.get(branch_id) and self.root_kv is not None:
            return self.root_kv
        return None

    def reset(self):
        self.root_hidden = None
        self.root_kv = None
        self.records = {}
