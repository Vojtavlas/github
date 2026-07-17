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
    tok.encode = lambda text, **kwargs: {"YES": [1], "NO": [2]}.get(text, [])
    return tok


def test_verifier_score_ratio():
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: {"YES": [1], "NO": [2]}.get(text, [])
    cgee = MagicMock()
    model = MagicMock()
    verifier = Verifier(model, tokenizer, cgee, cfg, device="cpu")

    logits = torch.tensor([1.0, 2.0, 3.0])
    score = verifier._verifier_score(logits)

    probs = torch.softmax(logits, dim=-1)
    expected = probs[1].item() / (probs[1].item() + probs[2].item() + 1e-10)
    assert score == pytest.approx(expected)


def test_verifier_score_multi_token():
    """A tokenizer may split YES/NO into several subtokens; all should be summed."""
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: {"YES": [3, 4], "NO": [5]}.get(text, [])
    verifier = Verifier(None, tokenizer, None, cfg, device="cpu")

    logits = torch.zeros(10)
    logits[3] = 1.0
    logits[4] = 5.0
    logits[5] = 2.0
    probs = torch.softmax(logits, dim=-1)
    expected = (probs[3] + probs[4]) / (probs[3] + probs[4] + probs[5])

    score = verifier._verifier_score(logits)
    assert score == pytest.approx(expected.item())


def test_verifier_score_missing_tokens():
    """If no YES/NO ids are present in the vocabulary, score should be neutral."""
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: []
    verifier = Verifier(None, tokenizer, None, cfg, device="cpu")

    logits = torch.randn(10)
    score = verifier._verifier_score(logits)
    assert score == pytest.approx(0.5)


def test_verifier_score_case_variants():
    """Lowercase, title-case and leading-space variants should contribute mass."""
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: {" yes": [6], " no": [7]}.get(text, [])
    verifier = Verifier(None, tokenizer, None, cfg, device="cpu")

    logits = torch.zeros(10)
    logits[6] = 5.0
    logits[7] = 1.0
    probs = torch.softmax(logits, dim=-1)
    expected = probs[6] / (probs[6] + probs[7])

    score = verifier._verifier_score(logits)
    assert score == pytest.approx(expected.item())


def test_verifier_score_2d_logits():
    """A 2-D (batch, vocab) logits tensor should be handled by taking row 0."""
    cfg = _config()
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, **kwargs: {"YES": [3, 4], "NO": [5]}.get(text, [])
    verifier = Verifier(None, tokenizer, None, cfg, device="cpu")

    logits_1d = torch.zeros(10)
    logits_1d[3] = 1.0
    logits_1d[4] = 5.0
    logits_1d[5] = 2.0
    logits_2d = logits_1d.unsqueeze(0)

    expected = verifier._verifier_score(logits_1d)
    score = verifier._verifier_score(logits_2d)
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
