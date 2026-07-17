from reasonflow import BranchResult, SolveResult


def test_branch_result_defaults():
    branch = BranchResult(
        branch_id=0,
        prompt="prompt",
        text="answer",
        full_text="full",
        generation_confidence=0.8,
    )
    assert branch.branch_id == 0
    assert branch.prompt == "prompt"
    assert branch.text == "answer"
    assert branch.full_text == "full"
    assert branch.generation_confidence == 0.8
    assert branch.verification_score == 0.0
    assert branch.verified is False
    assert branch.early_exit_layer is None


def test_solve_result_defaults():
    result = SolveResult(problem="2+2", best_text="4")
    assert result.problem == "2+2"
    assert result.best_text == "4"
    assert result.branches == []
    assert result.generation_time_ms == 0.0
    assert result.verification_time_ms == 0.0
    assert result.total_time_ms == 0.0
    assert result.skipped_verification is False
    assert not hasattr(result, "baseline_time_ms")
    assert not hasattr(result, "speedup")
