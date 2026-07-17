"""Unit tests for the autoregressive decoder."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache

from reasonflow.config import EngineConfig
from reasonflow.decoder import Decoder
from reasonflow.sampler import Sampler


class FakeModel:
    """Deterministic model that returns predictable logits and a real DynamicCache."""

    def __init__(
        self, vocab_size: int = 5, num_layers: int = 2, num_heads: int = 2, head_dim: int = 4
    ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        use_cache=False,
    ):
        B, S = input_ids.shape
        last = input_ids[:, -1]
        next_token = (last + 1) % self.vocab_size
        logits = torch.full(
            (B, S, self.vocab_size),
            -1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, :].scatter_(
            -1,
            next_token.unsqueeze(-1),
            torch.full((B, 1), 10.0, dtype=torch.float32, device=input_ids.device),
        )
        if not use_cache:
            return SimpleNamespace(logits=logits)

        if past_key_values is None:
            past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            k = torch.zeros(
                B, self.num_heads, S, self.head_dim, dtype=torch.float32, device=input_ids.device
            )
            v = torch.zeros_like(k)
            past_key_values.update(k, v, layer_idx)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


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
    sampler.sample.side_effect = [(torch.tensor([i]), torch.tensor([0.5])) for i in range(4)]

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


def test_decode_rejects_batch_size_not_one():
    cfg = _config(max_new_tokens=2)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99

    sampler = MagicMock()
    model = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    first_ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[1, 1], [1, 1]])

    with pytest.raises(ValueError, match="batch_size == 1"):
        decoder.decode(first_ids, None, mask, max_new_tokens=2)

    assert sampler.sample.call_count == 0
    assert model.call_count == 0


def test_continue_generate_rejects_batch_size_not_one():
    cfg = _config(max_new_tokens=2)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99

    sampler = MagicMock()
    model = MagicMock()

    decoder = Decoder(model, tokenizer, sampler, cfg)
    first_logits = torch.randn(2, 10)

    with pytest.raises(ValueError, match="batch_size == 1"):
        decoder.continue_generate(first_logits, MagicMock(), torch.tensor([[1, 1]]), 2)

    assert sampler.sample.call_count == 0
    assert model.call_count == 0


def test_decode_batch_greedy_equivalence():
    """Batched greedy decoding must match serial continue_generate on a FakeModel."""
    vocab_size = 5
    eos_token_id = 4
    pad_token_id = 0

    cfg = EngineConfig()
    cfg.max_new_tokens = 5
    cfg.temperature = 0.0
    cfg.top_p = 1.0
    cfg.max_seq_len = 20

    tokenizer = MagicMock()
    tokenizer.eos_token_id = eos_token_id
    tokenizer.pad_token_id = pad_token_id

    model = FakeModel(vocab_size=vocab_size)
    sampler = Sampler(cfg)
    decoder = Decoder(model, tokenizer, sampler, cfg)

    B = 3
    max_new_tokens = 5
    init_len = 3  # prefix + max suffix length (max_L)

    # Left-padded initial sequences with mixed real lengths.
    input_ids = torch.full((B, init_len), pad_token_id, dtype=torch.long)
    input_ids[0, init_len - 1] = 3  # real length 1 -> first token EOS
    input_ids[1, init_len - 2 :] = torch.tensor([1, 2])  # real length 2
    input_ids[2, :] = torch.tensor([0, 1, 2])  # real length 3

    attention_mask = torch.zeros(B, init_len + max_new_tokens, dtype=torch.long)
    attention_mask[:, :init_len] = (input_ids != pad_token_id).long()
    seq_lens = [(input_ids[i] != pad_token_id).sum().item() for i in range(B)]

    # Prefill once to produce the batched first logits and KV cache.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask[:, :init_len],
        use_cache=True,
    )
    first_logits = out.logits[:, -1, :]
    past_key_values = out.past_key_values

    batch_seqs, batch_confs = decoder.decode_batch(
        first_logits,
        past_key_values,
        attention_mask,
        max_new_tokens,
        seq_lens,
    )

    # Serial reference: run each row independently through continue_generate.
    serial_seqs = []
    serial_confs = []
    for i in range(B):
        row_mask = attention_mask[i : i + 1, :init_len]
        row_out = model(
            input_ids=input_ids[i : i + 1],
            attention_mask=row_mask,
            use_cache=True,
        )
        row_seq, row_conf, _ = decoder.continue_generate(
            row_out.logits[:, -1, :],
            row_out.past_key_values,
            row_mask,
            max_new_tokens,
        )
        serial_seqs.append(row_seq[0])
        serial_confs.append(row_conf)

    assert len(batch_seqs) == B
    assert len(batch_confs) == B
    for i in range(B):
        assert torch.equal(batch_seqs[i], serial_seqs[i]), f"row {i} token mismatch"
        assert abs(batch_confs[i] - serial_confs[i]) < 1e-5, f"row {i} confidence mismatch"

    # Per-row EOS: row 0 stops immediately, rows 1 and 2 continue.
    assert batch_seqs[0].numel() == 1
    assert batch_seqs[0][-1].item() == eos_token_id
    assert batch_seqs[1].numel() > batch_seqs[0].numel()
    assert batch_seqs[2].numel() > batch_seqs[0].numel()
    assert all(s.numel() <= max_new_tokens for s in batch_seqs)
    assert all((s[-1].item() == eos_token_id) for s in batch_seqs)
