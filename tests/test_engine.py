import os

import pytest

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
