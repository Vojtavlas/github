import os

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

    engine.branch_generator.generate = fake_generate.__get__(
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
