"""Tests for reasonflow.branch_and_share TailPiAdapter."""

import json
import threading
import time
from pathlib import Path

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchContext,
    StagnationConfig,
    TailPiAdapter,
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


def _write_lines(path: Path, events: list, delay: float = 0.01) -> None:
    time.sleep(0.05)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
            f.flush()
            time.sleep(delay)


def test_tail_adapter_replays_events_in_order(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text("")
    events = [
        {"kind": "tool_call", "name": "read", "args": {"path": "x.py"}},
        {"kind": "test", "name": "test_a", "passed": True, "output": ""},
        {"kind": "status", "status": "success"},
    ]

    writer = threading.Thread(target=_write_lines, args=(log, events))
    writer.start()

    try:
        adapter = TailPiAdapter(log, timeout_seconds=1.0, heartbeat_interval=0.005)
        control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.005)
        adapter.reset(_make_context(tmp_path))
        outcome = adapter.run(control)
    finally:
        writer.join()

    assert outcome.status.value == "success"
    assert control.monitor.tool_calls[0].name == "read"
    assert control.monitor.test_results[0].name == "test_a"


def test_tail_adapter_stops_on_status_event(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text("")
    events = [
        {"kind": "command", "command": "pytest", "output": "ok", "exit_code": 0},
        {"kind": "status", "status": "success", "result": "done"},
        {"kind": "discovery", "text": "ignored"},
    ]

    writer = threading.Thread(target=_write_lines, args=(log, events))
    writer.start()

    try:
        adapter = TailPiAdapter(log, timeout_seconds=1.0, heartbeat_interval=0.005)
        control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.005)
        adapter.reset(_make_context(tmp_path))
        outcome = adapter.run(control)
    finally:
        writer.join()

    assert outcome.status.value == "success"
    assert outcome.result == "done"
    assert len(control.monitor.discoveries) == 0


def test_tail_adapter_handles_truncation(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text("old1\nold2\n")

    def truncate_and_write() -> None:
        time.sleep(0.05)
        log.write_text("")
        time.sleep(0.05)
        events = [
            {"kind": "discovery", "text": "after rotation"},
            {"kind": "status", "status": "success"},
        ]
        with log.open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                f.flush()
                time.sleep(0.01)

    writer = threading.Thread(target=truncate_and_write)
    writer.start()

    try:
        adapter = TailPiAdapter(log, timeout_seconds=1.0, heartbeat_interval=0.005)
        control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.005)
        adapter.reset(_make_context(tmp_path))
        outcome = adapter.run(control)
    finally:
        writer.join()

    assert outcome.status.value == "success"
    assert control.monitor.discoveries[0] == "after rotation"


def test_tail_adapter_does_not_reread_existing_lines(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"kind": "discovery", "text": "existing"}) + "\n"
    )

    def append_status() -> None:
        time.sleep(0.05)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "status", "status": "success"}) + "\n")
            f.flush()

    writer = threading.Thread(target=append_status)
    writer.start()

    try:
        adapter = TailPiAdapter(log, timeout_seconds=1.0, heartbeat_interval=0.005)
        control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.005)
        adapter.reset(_make_context(tmp_path))
        outcome = adapter.run(control)
    finally:
        writer.join()

    assert outcome.status.value == "success"
    assert len(control.monitor.discoveries) == 0


def test_tail_adapter_times_out_on_empty_file(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text("")
    adapter = TailPiAdapter(log, timeout_seconds=0.05, heartbeat_interval=0.01)
    control = _make_control(timeout_seconds=0.05, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))
    outcome = adapter.run(control)

    assert outcome.status.value == "error"
    assert "timed out" in outcome.error.lower()


def test_tail_adapter_missing_file(tmp_path: Path) -> None:
    log = tmp_path / "does_not_exist.jsonl"
    adapter = TailPiAdapter(log, timeout_seconds=0.1, heartbeat_interval=0.01)
    control = _make_control(timeout_seconds=0.1, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))
    outcome = adapter.run(control)

    assert outcome.status.value == "error"
    assert "not found" in outcome.error.lower()
