from unittest.mock import MagicMock

import torch

from reasonflow.asks import (
    ASKSManager,
    CosineSimilarity,
    EuclideanSimilarity,
    ExponentialWeights,
    LinearWeights,
)
from reasonflow.config import RKSCConfig


def make_hidden_states(batch: int = 1, layers: int = 4, hidden: int = 16):
    return [torch.randn(batch, 5, hidden) for _ in range(layers)]


def make_stacked_hidden(layers: int = 4, hidden: int = 16):
    vecs = [torch.randn(hidden) for _ in range(layers)]
    stacked = torch.stack(vecs)
    norms = stacked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return stacked / norms


def test_cosine_similarity_identical():
    metric = CosineSimilarity()
    root = make_stacked_hidden()
    scores = metric.compute(root, root)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-6)


def test_cosine_similarity_different():
    metric = CosineSimilarity()
    root = make_stacked_hidden()
    branch = make_stacked_hidden()
    scores = metric.compute(root, branch)
    assert scores.abs().max() <= 1.0 + 1e-6
    assert not torch.allclose(scores, torch.ones_like(scores), atol=1e-3)


def test_euclidean_similarity_identical():
    metric = EuclideanSimilarity()
    root = make_stacked_hidden()
    scores = metric.compute(root, root)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-6)


def test_euclidean_similarity_decreases_with_distance():
    metric = EuclideanSimilarity()
    root = make_stacked_hidden()
    branch = root + 10.0
    scores = metric.compute(root, branch)
    assert scores.max() < 0.5


def test_exponential_weights_combine():
    strategy = ExponentialWeights()
    n = 4
    assert strategy.combine(torch.ones(n)) == 1.0
    low_early = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    low_late = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    assert strategy.combine(low_early) > strategy.combine(low_late)


def test_linear_weights_combine():
    strategy = LinearWeights()
    n = 4
    assert strategy.combine(torch.ones(n)) == 1.0
    low_early = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    low_late = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    assert strategy.combine(low_early) > strategy.combine(low_late)


def test_asks_manager_delegates_to_metric_and_weighting():
    cfg = RKSCConfig(tau=0.5)
    asks = ASKSManager(cfg, {})
    root_hs = make_hidden_states()
    asks.capture_root(None, root_hs)

    metric = MagicMock(spec=CosineSimilarity)
    metric.compute.return_value = torch.tensor([0.5, 0.6, 0.7, 0.8])
    weighting = MagicMock(spec=ExponentialWeights)
    weighting.combine.return_value = 0.9

    asks.metric = metric
    asks.weighting = weighting
    result = asks.score_branch(0, make_hidden_states())

    assert result is True
    metric.compute.assert_called_once()
    weighting.combine.assert_called_once()


def test_asks_manager_accepts_custom_metric_and_weighting():
    cfg = RKSCConfig(tau=0.5)
    metric = CosineSimilarity()
    weighting = LinearWeights()
    asks = ASKSManager(cfg, {}, metric=metric, weighting=weighting)
    assert asks.metric is metric
    assert asks.weighting is weighting


def test_asks_identical_branch_reuses():
    cfg = RKSCConfig(tau=0.75)
    asks = ASKSManager(cfg, {})
    root_hs = make_hidden_states()
    asks.capture_root(None, root_hs)
    # The exact same hidden states should have cosine similarity 1.0.
    assert asks.score_branch(0, root_hs) is True


def test_asks_dissimilar_branch_is_rejected():
    cfg = RKSCConfig(tau=0.99)
    asks = ASKSManager(cfg, {})
    root_hs = make_hidden_states()
    asks.capture_root(None, root_hs)
    branch_hs = [torch.randn_like(h) for h in root_hs]
    assert asks.score_branch(0, branch_hs) is False


def test_score_branches_partitions_rows():
    cfg = RKSCConfig(tau=0.5)
    asks = ASKSManager(cfg, {})
    root_hs = make_hidden_states()
    asks.capture_root(None, root_hs)

    B = 3
    layers = 4
    hidden = 16
    batched_hs = [torch.randn(B, 5, hidden) for _ in range(layers)]

    calls = []

    def fake_score_branch(branch_id, branch_hidden):
        calls.append((branch_id, branch_hidden))
        return branch_id % 2 == 0

    asks.score_branch = fake_score_branch

    branch_ids = [0, 1, 2]
    result = asks.score_branches(branch_ids, batched_hs)

    assert result == {0: True, 1: False, 2: True}
    assert len(calls) == 3
    for branch_id, branch_hidden in calls:
        assert len(branch_hidden) == layers
        for h in branch_hidden:
            assert h.shape == (1, 5, hidden)


def test_score_branches_empty_hidden_states_returns_all_false():
    cfg = RKSCConfig(tau=0.5)
    asks = ASKSManager(cfg, {})
    asks.capture_root(None, make_hidden_states())

    result = asks.score_branches([0, 1], [])
    assert result == {0: False, 1: False}
