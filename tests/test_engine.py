import os

import pytest
import torch

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer

SKIP_ENGINE = os.environ.get("SKIP_ENGINE_TESTS") == "1"


@pytest.fixture(scope="module")
def gpt2_engine():
    if SKIP_ENGINE:
        pytest.skip("Engine integration tests disabled")
    model, tokenizer = load_model_and_tokenizer("gpt2", device="cpu")
    cfg = EngineConfig(
        branching_factor=2, max_new_tokens=5, temperature=0.0, device="cpu"
    )
    return MultiBranchEngine(model, tokenizer, cfg)


def test_solve_runs(gpt2_engine):
    result = gpt2_engine.solve("What is 2 + 2?")
    assert result.best_text is not None
    assert len(result.branches) == 2
    assert result.total_time_ms > 0


def test_baseline_runs(gpt2_engine):
    result = gpt2_engine.baseline_solve("What is 2 + 2?")
    assert result.best_text is not None
    assert len(result.branches) == 2
    assert result.total_time_ms > 0


def test_prefix_kv_cache_not_mutated(gpt2_engine):
    """The shared prefix cache must survive multiple branch decodes unchanged."""
    problem = "What is 2 + 2?"
    prefix_str = gpt2_engine._shared_prefix(problem)
    inputs = gpt2_engine._tokenize(prefix_str)
    with torch.inference_mode():
        out = gpt2_engine.model(**inputs, use_cache=True)
    initial_len = out.past_key_values.get_seq_length()

    gpt2_engine.solve(problem)

    assert out.past_key_values.get_seq_length() == initial_len


def test_asks_gates_reuse(gpt2_engine):
    """ASKS should score every branch and allow prefix reuse for the shared prompt."""
    result = gpt2_engine.solve("What is 2 + 2?")
    assert len(gpt2_engine.asks.records) == gpt2_engine.config.branching_factor
    assert all(gpt2_engine.asks.records.values())
    assert result.total_time_ms > 0
