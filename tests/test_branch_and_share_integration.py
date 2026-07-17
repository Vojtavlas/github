"""Integration tests for branch_and_share adapters, launcher, store, and engine."""

import json
import sys
from pathlib import Path

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchAndShareEngine,
    BranchContext,
    BranchStartPoint,
    ExperiencePacket,
    ExperienceStore,
    FileStreamPiAdapter,
    MemoryBranchManager,
    MockPiAdapter,
    StagnationConfig,
    SubprocessPiAdapter,
)
from reasonflow.branch_and_share.control import TrajectoryControl
from reasonflow.branch_and_share.launcher import BranchSessionLauncher


def _write_event_log(path: Path, events: list) -> None:
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _make_control() -> TrajectoryControl:
    from reasonflow.branch_and_share.detector import StagnationDetector
    from reasonflow.branch_and_share.metrics import MetricsTracker
    from reasonflow.branch_and_share.monitor import TrajectoryMonitor

    config = BranchAndShareConfig(stagnation=StagnationConfig(repeat_threshold=2))
    return TrajectoryControl(
        TrajectoryMonitor(), StagnationDetector(config.stagnation), MetricsTracker(), config
    )


def test_file_stream_adapter_replays_events_and_returns_success(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    _write_event_log(
        log,
        [
            {"kind": "tool_call", "name": "read", "args": {"path": "x.py"}},
            {"kind": "test", "name": "test_a", "passed": True, "output": ""},
            {"kind": "status", "status": "success"},
        ],
    )
    adapter = FileStreamPiAdapter(str(log))
    control = _make_control()
    outcome = adapter.run(control)

    assert outcome.status.value == "success"
    assert len(control.monitor.tool_calls) == 1
    assert control.monitor.tool_calls[0].name == "read"
    assert len(control.monitor.test_results) == 1
    assert control.monitor.test_results[0].passed is True


def test_file_stream_adapter_detects_stagnation(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    _write_event_log(
        log,
        [
            {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
            {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
            {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
        ],
    )
    adapter = FileStreamPiAdapter(str(log))
    control = _make_control()
    outcome = adapter.run(control)

    assert outcome.status.value == "stagnation"
    assert "repeated_command" in (outcome.report.signals if outcome.report else [])


def test_subprocess_adapter_replays_events(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'kind': 'tool_call', 'name': 'read', 'args': {'path': 'x.py'}}))\n"
        "print(json.dumps({'kind': 'test', 'name': 'test_a', 'passed': True, 'output': ''}))\n"
        "print(json.dumps({'kind': 'status', 'status': 'success'}))\n"
    )
    adapter = SubprocessPiAdapter([sys.executable, str(script)])
    control = _make_control()
    adapter.reset(
        BranchContext(
            branch_id=0,
            worktree_path=str(tmp_path),
            start_ref="base",
            start_commit="base",
        )
    )
    outcome = adapter.run(control)

    assert outcome.status.value == "success"
    assert control.monitor.tool_calls[0].name == "read"
    assert control.monitor.test_results[0].name == "test_a"


def test_subprocess_adapter_returns_error_on_nonzero_exit(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import sys\n"
        "print('bad event', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    adapter = SubprocessPiAdapter([sys.executable, str(script)])
    control = _make_control()
    adapter.reset(
        BranchContext(
            branch_id=0,
            worktree_path=str(tmp_path),
            start_ref="base",
            start_commit="base",
        )
    )
    outcome = adapter.run(control)

    assert outcome.status.value == "error"
    assert outcome.error is not None
    assert "bad event" in outcome.error


def test_experience_store_roundtrip(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "experience.jsonl")
    packet = ExperiencePacket(
        files_and_symbols_inspected=[],
        commands_and_tests_run=["pytest -q"],
        modified_files_and_diff="",
        current_passing_tests=["test_a"],
        current_failing_tests=["test_b"],
        hypotheses_attempted=[],
        evidence_of_failure=[],
        useful_discoveries=["discovered"],
        recommended_next_actions=["fix it"],
        metrics=None,  # type: ignore[arg-type]
    )
    store.append(packet)
    loaded = store.load_recent(1)

    assert len(loaded) == 1
    assert loaded[0].current_passing_tests == ["test_a"]
    assert loaded[0].current_failing_tests == ["test_b"]
    assert loaded[0].recommended_next_actions == ["fix it"]
    assert store.load_all()[0].useful_discoveries == ["discovered"]


def test_branch_session_launcher_creates_branch_and_runs() -> None:
    branch_manager = MemoryBranchManager()
    config = BranchAndShareConfig(max_branches=2)
    scenario = [
        {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
        {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
        {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
    ]
    launcher = BranchSessionLauncher(
        config, branch_manager, lambda: MockPiAdapter(scenario=scenario)
    )
    context, outcome, control = launcher.launch(
        parent=None, branch_id=0, last_packet=None, start_point=BranchStartPoint.ORIGINAL
    )

    assert context.branch_id == 0
    assert outcome.status.value == "stagnation"
    assert len(control.monitor.commands) == 3


def test_engine_uses_launcher_and_store(tmp_path: Path) -> None:
    branch_manager = MemoryBranchManager()
    config = BranchAndShareConfig(max_branches=2, stagnation=StagnationConfig(repeat_threshold=2))
    scenarios = {
        0: [
            {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
            {"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1},
        ],
        1: [
            {"kind": "test", "name": "test_a", "passed": True, "output": ""},
        ],
    }
    store = ExperienceStore(tmp_path / "experience.jsonl")
    launcher = BranchSessionLauncher(
        config,
        branch_manager,
        lambda: MockPiAdapter(scenarios=scenarios, default_scenario=[]),
    )
    engine = BranchAndShareEngine(
        config, branch_manager, lambda: MockPiAdapter(), launcher=launcher, store=store
    )
    result = engine.solve()

    assert result.success is True
    assert len(result.branches) == 2
    assert len(store.load_all()) == 2
