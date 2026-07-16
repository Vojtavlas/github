import torch

from reasonflow.asks import ASKSManager
from reasonflow.config import RKSCConfig


def make_hidden_states(batch: int = 1, layers: int = 4, hidden: int = 16):
    return [torch.randn(batch, 5, hidden) for _ in range(layers)]


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
