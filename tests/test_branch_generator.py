"""Unit tests for the branch generator."""

from unittest.mock import MagicMock

import pytest
import torch

from reasonflow.branch_generator import BranchGenerator
from reasonflow.config import EngineConfig
from reasonflow.results import BranchResult


def _config(max_new_tokens: int = 3) -> EngineConfig:
    cfg = EngineConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.max_seq_len = 20
    cfg.device = "cpu"
    return cfg


def _make_tokenizer(input_ids: torch.Tensor):
    batch = MagicMock()
    batch.to.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    tok = MagicMock()
    tok.return_value = batch
    tok.decode = lambda ids, **kwargs: "dec"
    return tok


def test_generate_asks_reuses_prefix():
    cfg = _config(max_new_tokens=3)
    model = MagicMock()
    prefill_out = MagicMock()
    prefill_out.logits = torch.randn(1, 1, 10)
    prefill_out.past_key_values = MagicMock()
    prefill_out.hidden_states = [torch.randn(1, 5, 16)]
    model.return_value = prefill_out

    tokenizer = _make_tokenizer(torch.tensor([[4, 5]]))
    asks = MagicMock()
    asks.score_branch.return_value = True

    decoder = MagicMock()
    decoder.continue_generate.return_value = (torch.tensor([[6, 7]]), 0.9, MagicMock())

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")
    prefix_ids = torch.tensor([[1, 2, 3]])
    prefix_pkv = MagicMock()
    clone = MagicMock(return_value=MagicMock())
    gen.clone_kv_cache = clone

    result = gen.generate("prob", 0, prefix_ids, prefix_pkv, 3)

    assert isinstance(result, BranchResult)
    assert result.branch_id == 0
    assert result.generation_confidence == pytest.approx(0.9)
    clone.assert_called_once_with(prefix_pkv)
    asks.score_branch.assert_called_once()
    decoder.continue_generate.assert_called_once()


def test_generate_asks_rejected_uses_baseline():
    cfg = _config(max_new_tokens=3)
    model = MagicMock()
    prefill_out = MagicMock()
    prefill_out.logits = torch.randn(1, 1, 10)
    prefill_out.past_key_values = MagicMock()
    prefill_out.hidden_states = [torch.randn(1, 5, 16)]
    model.return_value = prefill_out

    tokenizer = _make_tokenizer(torch.tensor([[4, 5]]))
    asks = MagicMock()
    asks.score_branch.return_value = False

    decoder = MagicMock()
    # Baseline path tokenizes the full prompt to [[4, 5]] (2 tokens) and
    # decodes from scratch; return a sequence longer than the prompt so the
    # generated slice is non-empty.
    decoder.decode.return_value = (
        torch.tensor([[4, 5, 6, 7]]),
        0.7,
        MagicMock(),
    )

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")
    result = gen.generate("prob", 1, torch.tensor([[1, 2]]), MagicMock(), 2)

    assert isinstance(result, BranchResult)
    assert result.branch_id == 1
    assert result.generation_confidence == pytest.approx(0.7)
    asks.score_branch.assert_called_once()
    decoder.decode.assert_called_once()
    decoder.continue_generate.assert_not_called()


def test_generate_baseline_branch():
    cfg = _config(max_new_tokens=3)
    model = MagicMock()
    tokenizer = _make_tokenizer(torch.tensor([[1, 2, 3, 4]]))
    asks = MagicMock()
    decoder = MagicMock()
    # Prompt has 4 tokens, generate 2 -> sequence length 6.
    decoder.decode.return_value = (
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        0.6,
        MagicMock(),
    )

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")
    result = gen.generate_baseline_branch("prob", 0)

    assert isinstance(result, BranchResult)
    assert result.branch_id == 0
    assert result.text == "dec"
    assert result.generation_confidence == pytest.approx(0.6)
    decoder.decode.assert_called_once()


def test_generate_zero_max_new_tokens():
    cfg = _config(max_new_tokens=0)
    model = MagicMock()
    prefill_out = MagicMock()
    prefill_out.logits = torch.randn(1, 1, 10)
    prefill_out.past_key_values = MagicMock()
    prefill_out.hidden_states = [torch.randn(1, 5, 16)]
    model.return_value = prefill_out

    tokenizer = _make_tokenizer(torch.tensor([[4, 5]]))
    asks = MagicMock()
    asks.score_branch.return_value = True
    decoder = MagicMock()

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")
    result = gen.generate("prob", 0, torch.tensor([[1, 2, 3]]), MagicMock(), 3)

    assert isinstance(result, BranchResult)
    assert result.text == "dec"
    decoder.continue_generate.assert_not_called()
