import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reasonflow import EngineConfig, load_model_and_tokenizer
from reasonflow.results import BranchResult

SKIP_ENGINE = os.environ.get("SKIP_ENGINE_TESTS") == "1"

# The engine depends on cross-worktree modules (cache_adapter, model_adapter)
# that are not present in this worktree. If it cannot be imported, skip the
# integration tests rather than fail the whole suite.
try:
    from reasonflow.engine import MultiBranchEngine
    ENGINE_AVAILABLE = True
except ImportError as exc:
    MultiBranchEngine = None
    ENGINE_AVAILABLE = False
    ENGINE_IMPORT_ERROR = exc

FAKE_E2E_AVAILABLE = ENGINE_AVAILABLE and SKIP_ENGINE


@pytest.fixture(scope="module")
def qwen_engine():
    if not ENGINE_AVAILABLE:
        pytest.skip(f"Engine unavailable in this worktree: {ENGINE_IMPORT_ERROR}")
    if SKIP_ENGINE:
        pytest.skip("Engine integration tests disabled")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3.5-0.8B", device="cpu")
    cfg = EngineConfig(
        branching_factor=2, max_new_tokens=5, temperature=0.0, device="cpu"
    )
    return MultiBranchEngine(model, tokenizer, cfg)


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_solve_runs(qwen_engine):
    result = qwen_engine.solve("What is 2 + 2?")
    assert result.best_text is not None
    assert len(result.branches) == 2
    assert result.total_time_ms > 0


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_baseline_runs(qwen_engine):
    result = qwen_engine.baseline_solve("What is 2 + 2?")
    assert result.best_text is not None
    assert len(result.branches) == 2
    assert result.total_time_ms > 0


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_prefix_kv_cache_not_mutated(qwen_engine):
    """The shared prefix cache must survive multiple branch decodes unchanged."""
    problem = "What is 2 + 2?"
    prefix_str = qwen_engine._shared_prefix(problem)
    inputs = qwen_engine._tokenize(prefix_str)
    with pytest.importorskip("torch").inference_mode():
        out = qwen_engine.model(**inputs, use_cache=True)
    initial_len = out.past_key_values.get_seq_length()

    qwen_engine.solve(problem)

    assert out.past_key_values.get_seq_length() == initial_len


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_asks_gates_reuse(qwen_engine):
    """ASKS should score every branch and produce a boolean reuse decision."""
    result = qwen_engine.solve("What is 2 + 2?")
    assert len(qwen_engine.asks.records) == qwen_engine.config.branching_factor
    assert all(isinstance(v, bool) for v in qwen_engine.asks.records.values())
    assert result.total_time_ms > 0


@pytest.fixture
def mock_engine(monkeypatch):
    """Create a MultiBranchEngine backed by mocked model/tokenizer components."""
    if not ENGINE_AVAILABLE:
        pytest.skip(f"Engine unavailable in this worktree: {ENGINE_IMPORT_ERROR}")

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    hidden_dim = 8
    vocab_size = 16

    monkeypatch.setattr(
        "reasonflow.engine.get_transformer_layers", lambda model: [nn.Identity()]
    )

    class Batch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        eos_token_id = vocab_size - 1
        pad_token = "<pad>"
        pad_token_id = 0
        padding_side = "left"

        def __call__(
            self,
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ):
            ids = [ord(c) % vocab_size for c in text] or [0]
            return Batch(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )

        def encode(self, text, add_special_tokens=False):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return "".join(str(i) for i in ids)

    class FakeEmb:
        def __init__(self, weight):
            self.weight = weight

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.tensor([0.0]))

        def get_output_embeddings(self):
            return FakeEmb(torch.randn(vocab_size, hidden_dim))

        def forward(self, **kwargs):
            input_ids = kwargs.get("input_ids", torch.zeros(1, 1, dtype=torch.long))
            batch, seq = input_ids.shape[:2]

            class Out:
                past_key_values = None
                hidden_states = [torch.randn(batch, seq, hidden_dim)]
                logits = torch.randn(batch, seq, vocab_size)

            return Out()

    def fake_generate(self, problem, branch_id, prefix_ids, prefix_pkv, prefix_len):
        branch_hidden = [torch.zeros(1, hidden_dim)]
        self.asks.score_branch(branch_id, branch_hidden)
        return BranchResult(
            branch_id=branch_id,
            prompt="prompt",
            text=f"branch {branch_id}",
            full_text="full",
            generation_confidence=0.99,
        )

    def fake_generate_batch(self, problem, branch_ids, prefix_ids, prefix_pkv, prefix_len):
        return [
            fake_generate(self, problem, b, prefix_ids, prefix_pkv, prefix_len)
            for b in branch_ids
        ]

    def fake_baseline(self, problem, branch_id):
        return BranchResult(
            branch_id=branch_id,
            prompt="prompt",
            text=f"branch {branch_id}",
            full_text="full",
            generation_confidence=0.99,
        )

    tokenizer = FakeTokenizer()
    cfg = EngineConfig(
        branching_factor=2, max_new_tokens=1, temperature=0.0, device="cpu"
    )
    engine = MultiBranchEngine(FakeModel(), tokenizer, cfg)

    engine.branch_generator.generate_batch = fake_generate_batch.__get__(
        engine.branch_generator, type(engine.branch_generator)
    )
    engine.branch_generator.generate_baseline_branch = fake_baseline.__get__(
        engine.branch_generator, type(engine.branch_generator)
    )
    engine.cgee.should_skip_verification = lambda confs: True
    engine.verifier.verify = lambda problem, answer: (0.9, None, 0.0)

    return engine


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_solve_zero_branching_factor_raises(mock_engine):
    mock_engine.config.branching_factor = 0
    with pytest.raises(ValueError, match="branching_factor must be a positive integer"):
        mock_engine.solve("What is 2 + 2?")


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_baseline_solve_zero_branching_factor_raises(mock_engine):
    mock_engine.config.branching_factor = 0
    with pytest.raises(ValueError, match="branching_factor must be a positive integer"):
        mock_engine.baseline_solve("What is 2 + 2?")


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_solve_resets_asks_records(mock_engine):
    """Each solve() call starts with a clean asks.records dictionary."""
    mock_engine.config.branching_factor = 3
    mock_engine.solve("What is 2 + 2?")
    assert len(mock_engine.asks.records) == 3
    assert set(mock_engine.asks.records.keys()) == {0, 1, 2}

    # A subsequent solve with a different branching factor must not leak
    # branch ids from the previous run.
    mock_engine.config.branching_factor = 2
    mock_engine.solve("What is 3 + 3?")
    assert len(mock_engine.asks.records) == 2
    assert set(mock_engine.asks.records.keys()) == {0, 1}


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine adapters missing in worktree")
def test_solve_uses_generate_batch_when_batched(mock_engine, monkeypatch):
    """When use_batched_decoding is enabled, solve routes through generate_batch."""
    mock_engine.config.branching_factor = 3

    def fail_generate(*args, **kwargs):
        raise AssertionError("generate() should not be called when batching is enabled")

    monkeypatch.setattr(mock_engine.branch_generator, "generate", fail_generate)

    result = mock_engine.solve("What is 2 + 2?")
    assert len(result.branches) == 3
    assert [b.text for b in result.branches] == ["branch 0", "branch 1", "branch 2"]
    assert set(mock_engine.asks.records.keys()) == {0, 1, 2}


@pytest.fixture
def fake_e2e_engine(monkeypatch):
    """Build a deterministic MultiBranchEngine backed by FakeModel/FakeTokenizer."""
    if not ENGINE_AVAILABLE:
        pytest.skip(f"Engine unavailable in this worktree: {ENGINE_IMPORT_ERROR}")
    if not SKIP_ENGINE:
        pytest.skip("FakeModel end-to-end tests run only under SKIP_ENGINE_TESTS=1")

    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from transformers.cache_utils import DynamicCache

    class FakeEmb:
        def __init__(self, weight):
            self.weight = weight

    class FakeModel:
        def __init__(
            self,
            vocab_size: int = 64,
            hidden_dim: int = 16,
            num_layers: int = 1,
            num_heads: int = 2,
            head_dim: int = 4,
        ):
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.head_dim = head_dim

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def modules(self):
            return [self]

        def get_output_embeddings(self):
            return FakeEmb(
                torch.nn.Parameter(torch.randn(self.vocab_size, self.hidden_dim))
            )

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
            B, S = input_ids.shape[:2]
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
        def __init__(self, vocab_size: int = 64, word_ids=None):
            self.pad_token_id = 0
            self.eos_token_id = vocab_size - 1
            self.vocab_size = vocab_size
            self._word_to_id = dict(word_ids or {})
            self._id_to_word = {v: k for k, v in self._word_to_id.items()}
            self._next_id = 1
            if self._word_to_id:
                self._next_id = max(self._word_to_id.values()) + 1

        def _encode_word(self, word: str) -> int:
            if word not in self._word_to_id:
                while self._next_id in (self.pad_token_id, self.eos_token_id):
                    self._next_id += 1
                self._word_to_id[word] = self._next_id
                self._id_to_word[self._next_id] = word
                self._next_id += 1
            return self._word_to_id[word]

        def __call__(
            self,
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=None,
        ):
            tokens = text.split()
            ids = [self._encode_word(t) for t in tokens] or [self.pad_token_id]
            ids_tensor = torch.tensor([ids], dtype=torch.long)
            mask = torch.ones_like(ids_tensor)

            class Batch:
                def to(self, device):
                    return {"input_ids": ids_tensor, "attention_mask": mask}

            return Batch()

        def encode(self, text, add_special_tokens=False):
            return [self._encode_word(t) for t in text.split()] or [self.pad_token_id]

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            skip = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
            return " ".join(f"<{i}>" for i in ids if i not in skip)

    def make_engine(
        cfg=None,
        branch_hint=None,
        shared_prefix=None,
        word_ids=None,
        score_branches_fn=None,
        vocab_size: int = 64,
    ):
        if cfg is None:
            cfg = EngineConfig(
                use_batched_decoding=True,
                branching_factor=3,
                max_new_tokens=5,
                temperature=0.0,
                device="cpu",
            )

        tokenizer = FakeTokenizer(vocab_size=vocab_size, word_ids=word_ids)
        model = FakeModel(vocab_size=vocab_size, hidden_dim=16)

        monkeypatch.setattr(
            "reasonflow.engine.get_transformer_layers", lambda m: [nn.Identity()]
        )
        monkeypatch.setattr(
            "reasonflow.cgee.get_transformer_layers", lambda m: [nn.Identity()]
        )

        engine = MultiBranchEngine(model, tokenizer, cfg)

        monkeypatch.setattr(engine.asks, "score_branch", lambda bid, hs: True)
        if score_branches_fn is None:
            monkeypatch.setattr(
                engine.asks,
                "score_branches",
                lambda bids, hs: {bid: True for bid in bids},
            )
        else:
            monkeypatch.setattr(engine.asks, "score_branches", score_branches_fn)

        monkeypatch.setattr(
            engine.cgee, "should_skip_verification", lambda confs: True
        )
        monkeypatch.setattr(
            engine.verifier, "verify", lambda problem, answer: (0.9, None, 0.0)
        )

        if branch_hint is not None:
            monkeypatch.setattr(engine.branch_generator, "branch_hint", branch_hint)

        if shared_prefix is not None:
            prefix_fn = (
                shared_prefix
                if callable(shared_prefix)
                else lambda problem: shared_prefix
            )
            monkeypatch.setattr(engine, "_shared_prefix", prefix_fn)
            monkeypatch.setattr(engine.branch_generator, "shared_prefix", prefix_fn)

        return engine

    return make_engine


@pytest.mark.skipif(
    not FAKE_E2E_AVAILABLE,
    reason="FakeModel end-to-end tests run under SKIP_ENGINE_TESTS=1",
)
def test_batched_greedy_equivalence(fake_e2e_engine):
    """Batched solve produces the same greedy outputs as the serial path."""
    cfg_b = EngineConfig(
        use_batched_decoding=True,
        branching_factor=3,
        max_new_tokens=5,
        temperature=0.0,
        device="cpu",
    )
    cfg_s = EngineConfig(
        use_batched_decoding=False,
        branching_factor=3,
        max_new_tokens=5,
        temperature=0.0,
        device="cpu",
    )
    batched = fake_e2e_engine(cfg=cfg_b)
    serial = fake_e2e_engine(cfg=cfg_s)

    problem = "What is 2 + 2?"
    result_b = batched.solve(problem)
    result_s = serial.solve(problem)

    assert len(result_b.branches) == len(result_s.branches) == 3
    assert result_b.best_text == result_s.best_text
    for b_b, b_s in zip(result_b.branches, result_s.branches):
        assert b_b.branch_id == b_s.branch_id
        assert b_b.text == b_s.text


@pytest.mark.skipif(
    not FAKE_E2E_AVAILABLE,
    reason="FakeModel end-to-end tests run under SKIP_ENGINE_TESTS=1",
)
def test_mixed_hint_lengths(fake_e2e_engine):
    """Different token-length branch hints decode without shape errors."""
    hints = {0: "A", 1: "B C", 2: "D E F"}
    engine = fake_e2e_engine(branch_hint=lambda bid: hints[bid])

    result = engine.solve("What is 2 + 2?")
    assert len(result.branches) == 3
    assert all(isinstance(b.text, str) for b in result.branches)
    assert result.total_time_ms > 0


@pytest.mark.skipif(
    not FAKE_E2E_AVAILABLE,
    reason="FakeModel end-to-end tests run under SKIP_ENGINE_TESTS=1",
)
def test_per_row_eos(fake_e2e_engine):
    """One branch hits EOS early while another continues."""
    word_ids = {"A": 98, "X": 2, "Y": 3, "Z": 96, "prefix": 1}
    hints = {0: "A", 1: "X Y Z"}
    cfg = EngineConfig(
        use_batched_decoding=True,
        branching_factor=2,
        max_new_tokens=5,
        temperature=0.0,
        device="cpu",
    )

    engine = fake_e2e_engine(
        cfg=cfg,
        branch_hint=lambda bid: hints[bid],
        shared_prefix="prefix",
        word_ids=word_ids,
        vocab_size=100,
    )

    result = engine.solve("What is 2 + 2?")
    assert len(result.branches) == 2
    assert result.branches[0].text == ""
    assert len(result.branches[1].text.split()) == 2


@pytest.mark.skipif(
    not FAKE_E2E_AVAILABLE,
    reason="FakeModel end-to-end tests run under SKIP_ENGINE_TESTS=1",
)
def test_stable_branch_ordering(fake_e2e_engine):
    """Solve returns branches in branch_id order regardless of ASKS verdicts."""
    verdicts = {0: False, 1: True, 2: True}
    engine = fake_e2e_engine(
        score_branches_fn=lambda bids, hs: {bid: verdicts.get(bid, True) for bid in bids}
    )

    result = engine.solve("What is 2 + 2?")
    assert [b.branch_id for b in result.branches] == [0, 1, 2]


@pytest.mark.skipif(
    not FAKE_E2E_AVAILABLE,
    reason="FakeModel end-to-end tests run under SKIP_ENGINE_TESTS=1",
)
def test_asks_accept_reject_partitioning(fake_e2e_engine, monkeypatch):
    """Rejected rows fall back to baseline; approved rows are decoded in a batch."""
    verdicts = {0: True, 1: False, 2: True}
    engine = fake_e2e_engine(
        score_branches_fn=lambda bids, hs: {bid: verdicts.get(bid, True) for bid in bids}
    )

    orig_baseline = engine.branch_generator.generate_baseline_branch
    baseline_mock = MagicMock(side_effect=orig_baseline)
    monkeypatch.setattr(
        engine.branch_generator, "generate_baseline_branch", baseline_mock
    )

    orig_decode = engine.branch_generator.decoder.decode_batch
    decode_mock = MagicMock(side_effect=orig_decode)
    monkeypatch.setattr(engine.branch_generator.decoder, "decode_batch", decode_mock)

    result = engine.solve("problem")
    assert len(result.branches) == 3
    assert [b.branch_id for b in result.branches] == [0, 1, 2]

    baseline_mock.assert_called_once_with("problem", 1)
    assert decode_mock.call_count == 1

    first_logits, _, decode_mask, max_new, seq_lens = decode_mock.call_args.args
    assert first_logits.shape[0] == 2
    assert decode_mask.shape[0] == 2
    assert max_new == engine.config.max_new_tokens
    assert len(seq_lens) == 2
