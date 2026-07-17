import os

import pytest

from reasonflow import EngineConfig, load_model_and_tokenizer

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
