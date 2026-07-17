"""Attention-Similarity KV Sharing (ASKS)."""

import abc
from typing import Dict, List, Optional

import torch

from .utils import squeeze_hidden


class SimilarityMetric(abc.ABC):
    """Abstract base class for computing layer-wise hidden-state similarity."""

    @abc.abstractmethod
    def compute(self, root_hidden: torch.Tensor, branch_hidden: torch.Tensor) -> torch.Tensor:
        """Return a 1-D tensor of per-layer similarity scores.

        Args:
            root_hidden: ``[n_layers, hidden_dim]`` unit-normalised root states.
            branch_hidden: ``[n_layers, hidden_dim]`` branch states.

        Returns:
            A ``[n_layers]`` tensor with one score per layer. Higher values
            indicate greater similarity between root and branch states.
        """
        ...


class CosineSimilarity(SimilarityMetric):
    """Layer-wise cosine similarity between root and branch hidden states."""

    def compute(self, root_hidden: torch.Tensor, branch_hidden: torch.Tensor) -> torch.Tensor:
        dev = root_hidden.device
        branch = branch_hidden.to(dev)
        b_norms = branch.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_unit = branch / b_norms
        cos = (root_hidden.to(dev) * b_unit).sum(-1)
        return cos


class EuclideanSimilarity(SimilarityMetric):
    """Layer-wise RBF-style similarity from Euclidean distance.

    Identical (unit-normalised) vectors yield a similarity of ``1.0``;
    larger distances decay smoothly toward ``0.0``.
    """

    def compute(self, root_hidden: torch.Tensor, branch_hidden: torch.Tensor) -> torch.Tensor:
        dev = root_hidden.device
        branch = branch_hidden.to(dev)
        b_norms = branch.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_unit = branch / b_norms
        dist = (root_hidden.to(dev) - b_unit).norm(dim=-1)
        return torch.exp(-dist)


class WeightingStrategy(abc.ABC):
    """Abstract base class for combining per-layer scores into a scalar."""

    @abc.abstractmethod
    def combine(self, layer_scores: torch.Tensor) -> float:
        """Combine ``[n_layers]`` scores into a single float."""
        ...


class ExponentialWeights(WeightingStrategy):
    """Emphasise later layers using exponentially increasing weights."""

    def __init__(self, scale: float = 1.5):
        self.scale = scale

    def combine(self, layer_scores: torch.Tensor) -> float:
        n = layer_scores.shape[0]
        dev = layer_scores.device
        weights = torch.exp(torch.linspace(0.0, self.scale, n, device=dev))
        weights = weights / weights.sum()
        return float((layer_scores * weights).sum().item())


class LinearWeights(WeightingStrategy):
    """Emphasise later layers using linearly increasing weights."""

    def __init__(self, start: float = 1.0, end: float = 2.0):
        self.start = start
        self.end = end

    def combine(self, layer_scores: torch.Tensor) -> float:
        n = layer_scores.shape[0]
        dev = layer_scores.device
        weights = torch.linspace(self.start, self.end, n, device=dev)
        weights = weights / weights.sum()
        return float((layer_scores * weights).sum().item())


class ASKSManager:
    """Gates prefix KV reuse by layer-wise hidden-state similarity.

    ASKS strictly generalises token-exact prefix caching: a branch that is
    lexically identical to the root has similarity 1.0 and always reuses the
    cache; a branch that is only semantically close can also be accepted when
    its weighted similarity exceeds ``tau``.
    """

    def __init__(
        self,
        cfg,
        arch: dict,
        metric: Optional[SimilarityMetric] = None,
        weighting: Optional[WeightingStrategy] = None,
    ):
        self.cfg = cfg
        self.arch = arch
        self.metric = metric or CosineSimilarity()
        self.weighting = weighting or ExponentialWeights()
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
        """Weighted similarity between a branch and the root."""
        if self.root_hidden is None or not branch_hidden:
            return 0.0
        dev = self.root_hidden.device
        b = torch.stack([squeeze_hidden(h).to(dev) for h in branch_hidden])
        layer_scores = self.metric.compute(self.root_hidden, b)
        return self.weighting.combine(layer_scores)

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
