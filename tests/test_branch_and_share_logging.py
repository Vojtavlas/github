"""Tests for branch_and_share session logging and reporting."""

import json

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchAndShareEngine,
    BranchContext,
    BranchSessionLogger,
    BranchSessionReport,
    MemoryBranchManager,
    StagnationConfig,
    TrajectoryControl,
    TrajectoryMonitor,
    TrajectoryOutcome,
    TrajectoryRunner,
    TrajectoryStatus,
)
from reasonflow.branch_and_share.detector import StagnationDetector
from reasonflow.branch_and_share.metrics import MetricsTracker


def _small_config(**kwargs):
    return BranchAndShareConfig(
        max_branches=2,
        stagnation=StagnationConfig(
            repeat_threshold=3,
            file_read_threshold=3,
            test_window=2,
            tool_failure_threshold=2,
            churn_threshold=3,
            token_limit=100000,
            token_warn_fraction=0.85,
            window=10,
        ),
        **kwargs,
    )
def _small_config(**kwargs):
    defaults = {
        "max_branches": 2,
        "stagnation": StagnationConfig(
            repeat_threshold=3,
            file_read_threshold=3,
            test_window=2,
            tool_failure_threshold=2,
            churn_threshold=3,
            token_limit=100000,
            token_warn_fraction=0.85,
            window=10,
        ),
    }
    defaults.update(kwargs)
    return BranchAndShareConfig(**defaults)


def test_logger_creates_session_file_in_repo_root(tmp_path):
    logger = BranchSessionLogger(repo_root=str(tmp_path))
    assert logger.path is not None
    assert logger.path.relative_to(tmp_path)
    assert logger.path.parent.name == "sessions"
    assert logger.path.parent.parent.name == ".reasonflow"
    logger.close()


def test_logger_is_noop_for_missing_repo_root():
    logger = BranchSessionLogger(repo_root="/nonexistent/path/12345")
    assert logger.path is None
    # Should not raise.
    logger.log_event(branch_id=0, kind="test", elapsed_ms=1.0)
    logger.close()


def test_logger_writes_valid_json_lines(tmp_path):
    logger = BranchSessionLogger(repo_root=str(tmp_path))
    logger.log_event(
        branch_id=1,
        kind="tool_call",
        elapsed_ms=12.5,
        stagnation_signals=["repeated_command"],
        outcome="success",
        payload={"name": "read"},
    )
    logger.log_event(
        branch_id=1,
        kind="status",
        elapsed_ms=45.0,
        outcome="success",
        payload={"result": {"ok": True}},
    )
    logger.close()

    lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["branch_id"] == 1
    assert first["kind"] == "tool_call"
    assert first["elapsed_ms"] == 12.5
    assert first["stagnation_signals"] == ["repeated_command"]
    assert first["outcome"] == "success"
    assert first["payload"] == {"name": "read"}
    assert "timestamp" in first


def test_logger_context_manager(tmp_path):
    with BranchSessionLogger(repo_root=str(tmp_path)) as logger:
        logger.log_event(branch_id=0, kind="branch_start", elapsed_ms=0.0)
    assert logger._fh is None
    assert logger.path.read_text(encoding="utf-8").strip()


def test_trajectory_control_logs_events(tmp_path):
    logger = BranchSessionLogger(repo_root=str(tmp_path))
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(_small_config().stagnation)
    metrics = MetricsTracker()
    control = TrajectoryControl(
        monitor,
        detector,
        metrics,
        _small_config(),
        logger=logger,
        branch_id=7,
    )
    metrics.start()

    control.record_tool_call("read", {"path": "x.py"}, failed=False, tokens=10)
    control.record_file_read("x.py", ["foo"])
    control.record_command("pytest", "1 passed", 0)
    control.record_test_result("test_a", True)
    control.record_model_call(tokens=100)
    control.record_token_usage(5)
    control.record_file_change("x.py")
    control.record_hypothesis("h", "act", "expected")
    control.record_discovery("found it")

    metrics.stop()
    logger.close()

    raw = logger.path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    kinds = {line["kind"] for line in lines}
    assert kinds == {
        "tool_call",
        "file_read",
        "command",
        "test",
        "model_call",
        "token_usage",
        "file_change",
        "hypothesis",
        "discovery",
    }
    for line in lines:
        assert line["branch_id"] == 7
        assert "timestamp" in line
        assert "elapsed_ms" in line
        assert isinstance(line["stagnation_signals"], list)
        assert "payload" in line


def test_engine_creates_session_log_when_log_repo_root_set(tmp_path):
    class RecordingRunner(TrajectoryRunner):
        def reset(self, context: BranchContext) -> None:
            pass

        def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
            control.record_tool_call("read", {"path": "x.py"})
            control.record_test_result("test_a", True)
            return TrajectoryOutcome(status=TrajectoryStatus.SUCCESS, result={"ok": True})

    config = _small_config(max_branches=1, log_repo_root=str(tmp_path))
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, RecordingRunner)
    result = engine.solve()

    assert result.success is True
    assert len(result.branches) == 1

    session_files = list((tmp_path / ".reasonflow" / "sessions").glob("*.jsonl"))
    assert len(session_files) == 1
    raw = session_files[0].read_text(encoding="utf-8")
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    kinds = [line["kind"] for line in lines]
    assert "branch_start" in kinds
    assert "tool_call" in kinds
    assert "test" in kinds
    assert "status" in kinds
    status_line = [line for line in lines if line["kind"] == "status"][0]
    assert status_line["outcome"] == "success"


def test_engine_report_after_success():
    class SuccessRunner(TrajectoryRunner):
        def reset(self, context: BranchContext) -> None:
            pass

        def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
            return TrajectoryOutcome(status=TrajectoryStatus.SUCCESS, result={"ok": True})

    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, SuccessRunner)
    engine.solve()
    report = engine.report()

    assert isinstance(report, BranchSessionReport)
    assert report.final_success is True
    assert report.final_outcome == TrajectoryStatus.SUCCESS
    assert len(report.branches) == 1
    assert report.total_time_ms >= 0
    assert report.pass_count == 0
    assert report.fail_count == 0


def test_engine_does_not_create_log_by_default(tmp_path, monkeypatch):
    """MemoryBranchManager with no log_repo_root should not write a session log."""
    class SuccessRunner(TrajectoryRunner):
        def reset(self, context: BranchContext) -> None:
            pass

        def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
            return TrajectoryOutcome(status=TrajectoryStatus.SUCCESS, result={"ok": True})

    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, SuccessRunner)
    result = engine.solve()

    assert result.success is True
    # No .reasonflow directory should appear in tmp_path unless explicitly set.
    assert not (tmp_path / ".reasonflow").exists()


def test_logger_stays_inside_repo_root(tmp_path):
    """The resolved session path must remain inside the provided repo root."""
    logger = BranchSessionLogger(repo_root=str(tmp_path))
    assert logger.path is not None
    assert str(logger.path).startswith(str(tmp_path.resolve()))
    logger.close()
