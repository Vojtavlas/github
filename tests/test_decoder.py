"""Unit tests for the autoregressive decoder."""

from unittest.mock import MagicMock

import torch

from reasonflow.config import EngineConfig
from reasonflow.decoder import Decoder


def _config(max_new_tokens: int = 5) -> EngineConfig:
    cfg = EngineConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.temperature = 0.0
    cfg.top_p = 1.0
    cfg.max_seq_len = 20
    return cfg


def test_decode_stops_at_eos():
    cfg = _config(max_new_tokens=5)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 5

    sampler = MagicMock()
    sampler.sample.side_effect = [
        (torch.tensor([2]), torch.tensor([0.8])),
        (torch.tensor([3]), torch.tensor([0.7])),
        (torch.tensor([5]), torch.tensor([0.9])),  # EOS
    ]

    model = MagicMock()
    model.return_value.past_key_values = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    first_ids = torch.tensor([[1, 2]])
    mask = torch.tensor([[1, 1]])
    seq, conf, _ = decoder.decode(first_ids, None, mask, max_new_tokens=5)

    assert seq.shape == (1, 5)  # 2 prefix + 3 generated tokens
    assert 0.0 < conf <= 1.0
    assert sampler.sample.call_count == 3


def test_decode_respects_max_new_tokens():
    cfg = _config(max_new_tokens=4)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99

    sampler = MagicMock()
    sampler.sample.side_effect = [
        (torch.tensor([i]), torch.tensor([0.5])) for i in range(4)
    ]

    model = MagicMock()
    model.return_value.past_key_values = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    first_ids = torch.tensor([[1]])
    mask = torch.tensor([[1]])
    seq, conf, _ = decoder.decode(first_ids, None, mask, max_new_tokens=4)

    assert seq.shape == (1, 5)  # 1 prefix + 4 generated tokens
    assert sampler.sample.call_count == 4


def test_continue_generate_from_first_logits():
    cfg = _config(max_new_tokens=3)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99

    sampler = MagicMock()
    sampler.sample.side_effect = [
        (torch.tensor([7]), torch.tensor([0.8])),
        (torch.tensor([8]), torch.tensor([0.7])),
        (torch.tensor([9]), torch.tensor([0.6])),
    ]

    model = MagicMock()
    model.return_value.past_key_values = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    first_logits = torch.randn(1, 10)
    mask = torch.tensor([[1, 1]])
    generated_ids, conf, _ = decoder.continue_generate(first_logits, MagicMock(), mask, 3)

    assert generated_ids.shape == (1, 3)
    assert sampler.sample.call_count == 3


def test_continue_generate_stops_at_eos():
    cfg = _config(max_new_tokens=5)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 5

    sampler = MagicMock()
    sampler.sample.side_effect = [
        (torch.tensor([2]), torch.tensor([0.8])),
        (torch.tensor([5]), torch.tensor([0.9])),  # EOS
    ]

    model = MagicMock()
    model.return_value.past_key_values = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    generated_ids, conf, _ = decoder.continue_generate(
        torch.randn(1, 10), MagicMock(), torch.tensor([[1, 1]]), 5
    )

    assert generated_ids.shape == (1, 2)
    assert sampler.sample.call_count == 2
