import torch

from reasonflow.cgee import CGEEAnalyzer
from reasonflow.config import RKSCConfig


def test_entropy_computation():
    cfg = RKSCConfig()
    unembed = torch.randn(100, 16)
    analyzer = CGEEAnalyzer(cfg, unembed, 4)
    logits = torch.randn(2, 100)
    ent = analyzer._entropy(logits)
    assert ent.shape == (2,)
    assert torch.all(ent >= 0)


def test_skip_verification_relative_gap():
    cfg = RKSCConfig(gen_conf_threshold=0.70, relative_gap_threshold=0.10)
    analyzer = CGEEAnalyzer(cfg, torch.randn(10, 16), 2)
    assert analyzer.should_skip_verification([0.9, 0.7]) is True
    assert analyzer.should_skip_verification([0.9, 0.85]) is False
    assert analyzer.should_skip_verification([0.5, 0.3]) is False
