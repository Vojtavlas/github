"""Unit tests for the verifier."""

from unittest.mock import MagicMock

import pytest
import torch

from reasonflow.config import EngineConfig
from reasonflow.verifier import Verifier


def _config() -> EngineConfig:
    cfg = EngineConfig()
    cfg.max_seq_len = 20
    cfg.device = "cpu"
    return cfg


def _make_tokenizer():
    batch = MagicMock()
    batch.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tok = MagicMock()
    tok.return_value = batch
    tok.encode = lambda text, **kwargs: {"YES": [1], "NO": [2]}[text]
    return tok


def test_verifier_score_ratio():
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: [1] if text == "YES" else [2]
    cgee = MagicMock()
    model = MagicMock()
    verifier = Verifier(model, tokenizer, cgee, cfg, device="cpu")

    logits = torch.tensor([1.0, 2.0, 3.0])
    score = verifier._verifier_score(logits)

    probs = torch.softmax(logits, dim=-1)
    expected = probs[1].item() / (probs[1].item() + probs[2].item() + 1e-10)
    assert score == pytest.approx(expected)


def test_verify_returns_score_and_exit_layer():
    cfg = _config()
    tokenizer = _make_tokenizer()
    cgee = MagicMock()
    cgee.analyze.return_value = (torch.randn(1, 10), 3, [0.1])
    model = MagicMock()
    verifier = Verifier(model, tokenizer, cgee, cfg, device="cpu")

    score, exit_layer, verify_ms = verifier.verify("What is 2+2?", "4")

    assert 0.0 <= score <= 1.0
    assert exit_layer == 3
    assert verify_ms >= 0
    cgee.analyze.assert_called_once()
