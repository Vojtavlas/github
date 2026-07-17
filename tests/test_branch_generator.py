"""Unit tests for the branch generator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache

from reasonflow.branch_generator import BranchGenerator
from reasonflow.config import EngineConfig
from reasonflow.decoder import Decoder
from reasonflow.results import BranchResult
from reasonflow.sampler import Sampler


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


class FakeModel:
    """Deterministic model that returns predictable logits and a real DynamicCache."""

    def __init__(
        self,
        vocab_size: int = 20,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int = 4,
        hidden_dim: int = 16,
    ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def modules(self):
        return [self]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        use_cache=False,
        output_hidden_states=False,
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
            out = SimpleNamespace(logits=logits)
            if output_hidden_states:
                out.hidden_states = [
                    torch.randn(B, S, self.hidden_dim, device=input_ids.device)
                    for _ in range(self.num_layers)
                ]
            return out

        if past_key_values is None:
            past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            k = torch.zeros(
                B,
                self.num_heads,
                S,
                self.head_dim,
                dtype=torch.float32,
                device=input_ids.device,
            )
            v = torch.zeros_like(k)
            past_key_values.update(k, v, layer_idx)

        out = SimpleNamespace(logits=logits, past_key_values=past_key_values)
        if output_hidden_states:
            out.hidden_states = [
                torch.randn(B, S, self.hidden_dim, device=input_ids.device)
                for _ in range(self.num_layers)
            ]
        return out


class FakeTokenizer:
    """Minimal whitespace tokenizer that assigns ids deterministically."""

    def __init__(self, vocab_size: int = 20):
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.vocab_size = vocab_size
        self._word_to_id: dict = {}
        self._id_to_word: dict = {}
        self._next_id = 1

    def _encode_word(self, word: str) -> int:
        if word not in self._word_to_id:
            while self._next_id in (self.pad_token_id, self.eos_token_id):
                self._next_id += 1
            self._word_to_id[word] = self._next_id
            self._id_to_word[self._next_id] = word
            self._next_id += 1
        return self._word_to_id[word]

    def __call__(
        self, text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=None
    ):
        tokens = text.split()
        ids = [self._encode_word(t) for t in tokens]
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(ids_tensor)

        class Batch:
            def to(self, device):
                return {"input_ids": ids_tensor, "attention_mask": mask}

        return Batch()

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        skip = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        words = []
        for i in ids:
            if i in skip:
                continue
            words.append(self._id_to_word.get(i, f"<{i}>"))
        return " ".join(words)


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


def test_generate_batch_partitions_asks():
    cfg = _config(max_new_tokens=3)
    cfg.temperature = 0.0

    model = FakeModel(vocab_size=20)
    tokenizer = FakeTokenizer(vocab_size=20)
    asks = MagicMock()
    asks.score_branches.return_value = {10: True, 20: False, 30: True}

    decoder = MagicMock()
    decoder.decode_batch.return_value = (
        [torch.tensor([6, 7, 8]), torch.tensor([9, 10, 11])],
        [0.9, 0.85],
    )

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")
    gen.branch_hint = lambda bid: "A"
    gen.generate_baseline_branch = MagicMock(
        return_value=BranchResult(
            branch_id=20,
            prompt="baseline prompt",
            text="baseline",
            full_text="baseline full",
            generation_confidence=0.5,
        )
    )

    prefix_ids = torch.tensor([[5]])
    prefix_out = model(prefix_ids, use_cache=True, output_hidden_states=True)
    prefix_pkv = prefix_out.past_key_values
    prefix_len = 1

    results = gen.generate_batch("prob", [10, 20, 30], prefix_ids, prefix_pkv, prefix_len)

    assert len(results) == 3
    assert [r.branch_id for r in results] == [10, 20, 30]
    assert decoder.decode_batch.call_count == 1
    assert gen.generate_baseline_branch.call_count == 1
    gen.generate_baseline_branch.assert_called_once_with("prob", 20)

    batch_first_logits, batch_cache, batch_mask, batch_max_new, batch_seq_lens = (
        decoder.decode_batch.call_args.args
    )
    assert batch_first_logits.shape[0] == 2
    assert batch_mask.shape[0] == 2
    assert batch_mask.shape[1] == prefix_len + 1 + cfg.max_new_tokens
    assert len(batch_seq_lens) == 2
    assert batch_seq_lens == [prefix_len + 1, prefix_len + 1]
    asks.score_branches.assert_called_once()


def test_generate_batch_mixed_hints():
    cfg = _config(max_new_tokens=4)
    cfg.temperature = 0.0

    model = FakeModel(vocab_size=20)
    tokenizer = FakeTokenizer(vocab_size=20)
    asks = MagicMock()
    asks.score_branch.return_value = True
    asks.score_branches.return_value = {5: True, 6: True, 7: True}

    sampler = Sampler(cfg)
    decoder = Decoder(model, tokenizer, sampler, cfg)

    gen = BranchGenerator(model, tokenizer, asks, decoder, cfg, device="cpu")

    def _hint(bid: int) -> str:
        return {5: "A", 6: "B C", 7: "D E F"}[bid]

    gen.branch_hint = _hint
    gen.generate_baseline_branch = MagicMock()

    prefix_ids = torch.tensor([[10, 11]])
    prefix_out = model(prefix_ids, use_cache=True, output_hidden_states=True)
    prefix_pkv = prefix_out.past_key_values
    prefix_len = 2

    results = gen.generate_batch("prob", [5, 6, 7], prefix_ids, prefix_pkv, prefix_len)

    assert len(results) == 3
    assert [r.branch_id for r in results] == [5, 6, 7]
    gen.generate_baseline_branch.assert_not_called()

    for i, bid in enumerate([5, 6, 7]):
        serial = gen.generate("prob", bid, prefix_ids, prefix_pkv, prefix_len)
        assert results[i].text == serial.text
