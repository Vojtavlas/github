"""Tests for the branch_and_share failure-aware branching layer."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchAndShareEngine,
    BranchContext,
    BranchMetrics,
    BranchStartPoint,
    ExperiencePacketBuilder,
    GitWorktreeBranchManager,
    MemoryBranchManager,
    MockPiAdapter,
    StagnationConfig,
    StagnationDetector,
    TrajectoryMonitor,
)


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("hello")
    (path / "foo.py").write_text("original")
    subprocess.run(["git", "add", "."], cwd=str(path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )


def _small_config() -> BranchAndShareConfig:
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
    )


def test_monitor_records_and_counts_duplicates():
    monitor = TrajectoryMonitor()
    for _ in range(3):
        monitor.record_tool_call("read_file", {"path": "x.py"})
    for _ in range(3):
        monitor.record_file_read("y.py", ["foo"])

    assert monitor.duplicated_work(window=10) == 4
    passing, failing, total = monitor.test_progress()
    assert passing == failing == total == 0

    monitor.record_test_result("test_a", True)
    monitor.record_test_result("test_b", False)
    monitor.record_test_result("test_a", False)
    passing, failing, total = monitor.test_progress()
    assert passing == 0
    assert failing == 2
    assert total == 2


def test_detector_repeated_command():
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(StagnationConfig(repeat_threshold=3, window=5))
    assert detector.check(monitor) is None

    for _ in range(3):
        monitor.record_command("pytest -q")
    report = detector.check(monitor)
    assert report is not None
    assert "repeated_command" in report.signals


def test_detector_repeated_file_read():
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(
        StagnationConfig(file_read_threshold=3, window=5)
    )
    for _ in range(3):
        monitor.record_file_read("same_file.py")
    report = detector.check(monitor)
    assert report is not None
    assert "repeated_file_read" in report.signals


def test_detector_repeated_tool_failure():
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(
        StagnationConfig(tool_failure_threshold=2, window=5)
    )
    monitor.record_tool_call("bash", {"cmd": "bad"}, failed=True)
    monitor.record_tool_call("bash", {"cmd": "bad"}, failed=True)
    report = detector.check(monitor)
    assert report is not None
    assert "repeated_tool_failure" in report.signals


def test_detector_no_test_improvement():
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(
        StagnationConfig(test_window=2, window=5)
    )
    monitor.record_test_result("t1", False, "err")
    monitor.record_test_result("t1", False, "err")
    report = detector.check(monitor)
    assert report is not None
    assert "no_test_improvement" in report.signals


def test_detector_token_limit():
    monitor = TrajectoryMonitor()
    detector = StagnationDetector(
        StagnationConfig(token_limit=100, token_warn_fraction=0.9)
    )
    monitor.record_token_usage(95)
    report = detector.check(monitor)
    assert report is not None
    assert "approaching_token_limit" in report.signals


def test_memory_branch_manager_creates_branches_and_checkpoints():
    manager = MemoryBranchManager(base_commit="base")
    ctx0 = manager.create_branch(None, BranchStartPoint.ORIGINAL, None, 0)
    assert ctx0.start_ref == "base"
    sha = manager.checkpoint(ctx0, "stuck")
    assert sha == "sha-0-stuck"

    ctx1 = manager.create_branch(
        ctx0, BranchStartPoint.LAST_CHECKPOINT, None, 1
    )
    assert ctx1.start_ref == sha
    assert ctx1.parent_branch_id == 0


def test_experience_packet_grounded_in_git(tmp_path):
    _init_git_repo(tmp_path)
    target = tmp_path / "foo.py"
    target.write_text("print(1)")

    monitor = TrajectoryMonitor(git_repo_root=str(tmp_path))
    monitor.record_test_result("test_foo", False, "assertion failed")
    monitor.record_command("pytest -q", "1 failed", 1)

    packet = ExperiencePacketBuilder(
        monitor,
        BranchContext(
            branch_id=0,
            worktree_path=str(tmp_path),
            start_ref="base",
            start_commit="base",
        ),
        BranchMetrics(),
    ).build()

    assert "foo.py" in packet.modified_files_and_diff
    assert "test_foo" in packet.current_failing_tests
    assert any("pytest" in s for s in packet.commands_and_tests_run)
    assert any("test_foo" in s for s in packet.evidence_of_failure)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_worktree_branch_manager(tmp_path):
    _init_git_repo(tmp_path)
    manager = GitWorktreeBranchManager(
        repo_root=str(tmp_path), worktrees_dir="worktrees"
    )
    ctx = manager.create_branch(
        None, BranchStartPoint.ORIGINAL, None, 0
    )

    assert os.path.isdir(ctx.worktree_path)
    assert (Path(ctx.worktree_path) / ".git").exists()

    current = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ctx.worktree_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert current == "rf-attempt-0"

    (Path(ctx.worktree_path) / "new.py").write_text("x = 1")
    sha = manager.checkpoint(ctx, "test-checkpoint")
    assert len(sha) == 40


def test_engine_branches_on_stagnation_then_succeeds():
    config = _small_config()
    manager = MemoryBranchManager()
    scenarios = {
        0: [
            {"kind": "file_read", "path": "stuck.py"},
            {"kind": "file_read", "path": "stuck.py"},
            {"kind": "file_read", "path": "stuck.py"},
        ],
        1: [
            {"kind": "test", "name": "test_ok", "passed": True},
        ],
    }
    engine = BranchAndShareEngine(
        config, manager, lambda: MockPiAdapter(scenarios=scenarios)
    )
    result = engine.solve()

    assert result.success
    assert result.metrics.branch_count == 2
    assert len(result.branches) == 2
    assert result.branches[1].start_ref == "sha-0-stuck-0"
    assert result.final_packet is not None
    assert result.final_packet.recommended_next_actions


def test_engine_respects_max_branches():
    config = _small_config()
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(
        config,
        manager,
        lambda: MockPiAdapter(
            scenario=[{"kind": "file_read", "path": "stuck.py"}] * 3
        ),
    )
    result = engine.solve()

    assert not result.success
    assert result.metrics.branch_count == 2
    assert result.best_branch_id is None
