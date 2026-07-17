"""Unit tests for the next-token sampler."""

from unittest.mock import patch

import pytest
import torch

from reasonflow.config import EngineConfig
from reasonflow.sampler import Sampler


def _config(temperature: float = 0.0, top_p: float = 1.0) -> EngineConfig:
    cfg = EngineConfig()
    cfg.temperature = temperature
    cfg.top_p = top_p
    return cfg


def test_greedy_argmax():
    cfg = _config(temperature=0.0)
    sampler = Sampler(cfg)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    token_id, conf = sampler.sample(logits)
    assert token_id.item() == 2
    assert 0.0 <= conf.item() <= 1.0


def test_temperature_sampling_uses_multinomial():
    cfg = _config(temperature=0.5)
    sampler = Sampler(cfg)
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    with patch("torch.multinomial") as mock_multi:
        mock_multi.return_value = torch.tensor([[1]])
        token_id, conf = sampler.sample(logits)
    assert token_id.item() == 1
    assert conf.item() > 0.0
    mock_multi.assert_called_once()


def test_top_p_filters_low_probability_tail():
    cfg = _config(temperature=1.0, top_p=0.5)
    sampler = Sampler(cfg)
    # Token 0 dominates the distribution.
    logits = torch.tensor([[10.0, 0.0, 0.0]])

    def side_effect(probs, **_):
        # After filtering and renormalization token 0 should have prob ~1.
        assert probs[0, 0].item() == pytest.approx(1.0, abs=1e-6)
        return torch.tensor([[0]])

    with patch("torch.multinomial") as mock_multi:
        mock_multi.side_effect = side_effect
        token_id, conf = sampler.sample(logits)
    assert token_id.item() == 0


def test_negative_temperature_raises():
    cfg = _config(temperature=-0.5)
    sampler = Sampler(cfg)
    with pytest.raises(ValueError):
        sampler.sample(torch.tensor([[1.0, 2.0, 3.0]]))


def test_very_small_positive_temperature_does_not_overflow():
    cfg = _config(temperature=1e-8, top_p=1.0)
    sampler = Sampler(cfg)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    token_id, conf = sampler.sample(logits)
    assert torch.isfinite(conf).all()
    assert token_id.item() == logits.argmax(dim=-1).item()


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.5, 2.0])
def test_top_p_out_of_range_raises(top_p):
    cfg = _config(temperature=1.0, top_p=top_p)
    sampler = Sampler(cfg)
    with pytest.raises(ValueError):
        sampler.sample(torch.tensor([[1.0, 2.0, 3.0]]))


def test_top_p_one_equals_no_filtering():
    cfg = _config(temperature=1.0, top_p=1.0)
    sampler = Sampler(cfg)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    expected_probs = torch.softmax(logits, dim=-1)

    def side_effect(probs, **_):
        assert torch.allclose(probs, expected_probs, atol=1e-6)
        return torch.tensor([[2]])

    with patch("torch.multinomial") as mock_multi:
        mock_multi.side_effect = side_effect
        token_id, conf = sampler.sample(logits)
    assert token_id.item() == 2
