"""Adversarial tests for reasonflow.branch_and_share SubprocessPiAdapter."""

import sys
from pathlib import Path

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchContext,
    FileStreamPiAdapter,
    StagnationConfig,
    SubprocessPiAdapter,
)
from reasonflow.branch_and_share.control import TrajectoryControl
from reasonflow.branch_and_share.detector import StagnationDetector
from reasonflow.branch_and_share.metrics import MetricsTracker
from reasonflow.branch_and_share.monitor import TrajectoryMonitor


def _make_control(**config_kwargs: object) -> TrajectoryControl:
    config = BranchAndShareConfig(
        stagnation=StagnationConfig(repeat_threshold=2),
        **config_kwargs,
    )
    return TrajectoryControl(
        TrajectoryMonitor(),
        StagnationDetector(config.stagnation),
        MetricsTracker(),
        config,
    )


def _make_context(tmp_path: Path) -> BranchContext:
    return BranchContext(
        branch_id=0,
        worktree_path=str(tmp_path),
        start_ref="base",
        start_commit="base",
    )


def test_subprocess_adapter_times_out(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, time\n"
        "time.sleep(2)\n"
        "print(json.dumps({'kind': 'status', 'status': 'success'}))\n"
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=0.1,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=0.1, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))
    outcome = adapter.run(control)

    assert outcome.status.value == "error"
    assert "timed out" in outcome.error.lower()


def test_subprocess_adapter_uses_config_timeouts(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text("import json, time\ntime.sleep(2)\n")
    # No explicit timeout on adapter: must fall back to control.config values.
    adapter = SubprocessPiAdapter([sys.executable, str(script)])
    control = _make_control(timeout_seconds=0.15, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))
    outcome = adapter.run(control)

    assert outcome.status.value == "error"
    assert "timed out" in outcome.error.lower()


def test_subprocess_adapter_streaming_with_slow_events(tmp_path: Path) -> None:
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, time\n"
        "print(json.dumps({'kind': 'tool_call', 'name': 'read', 'args': {'path': 'x.py'}}))\n"
        "time.sleep(0.05)\n"
        "print(json.dumps({'kind': 'test', 'name': 'test_a', 'passed': True, 'output': ''}))\n"
        "time.sleep(0.05)\n"
        "print(json.dumps({'kind': 'status', 'status': 'success'}))\n"
    )
    adapter = SubprocessPiAdapter([sys.executable, str(script)])
    control = _make_control()
    adapter.reset(_make_context(tmp_path))
    outcome = adapter.run(control)

    assert outcome.status.value == "success"
    assert control.monitor.tool_calls[0].name == "read"
    assert control.monitor.test_results[0].name == "test_a"


def test_file_stream_adapter_uses_event_stream_reader(tmp_path: Path) -> None:
    """Events that are not newline-terminated must still be replayed."""
    log = tmp_path / "events.jsonl"
    log.write_text(
        "{\"kind\": \"tool_call\", \"name\": \"read\", \"args\": {}}"
    )
    adapter = FileStreamPiAdapter(str(log))
    control = _make_control()
    outcome = adapter.run(control)

    assert outcome.status.value == "success"
    assert control.monitor.tool_calls[0].name == "read"
