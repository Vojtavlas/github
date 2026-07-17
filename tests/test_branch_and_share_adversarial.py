"""Adversarial tests for reasonflow.branch_and_share adapters."""

import sys
import time
from pathlib import Path

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchContext,
    FileStreamPiAdapter,
    StagnationConfig,
    SubprocessPiAdapter,
    TrajectoryStatus,
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


def _write_script(tmp_path: Path, name: str, code: str) -> Path:
    script = tmp_path / name
    script.write_text(code, encoding="utf-8")
    return script


# ---------------------------------------------------------------------------
# FileStreamPiAdapter adversarial cases
# ---------------------------------------------------------------------------


def test_file_stream_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    adapter = FileStreamPiAdapter(missing)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "Event log not found" in outcome.error
    assert str(missing) in outcome.error


def test_file_stream_empty_file(tmp_path: Path) -> None:
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert outcome.result == {"message": "completed"}


def test_file_stream_no_trailing_newline(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"kind": "tool_call", "name": "read", "args": {}}',
        encoding="utf-8",
    )
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert control.monitor.tool_calls[0].name == "read"


def test_file_stream_crlf_lines(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_bytes(
        b'{"kind": "test", "name": "t", "passed": true}\r\n'
        b'{"kind": "status", "status": "success", "result": "done"}\r\n'
    )
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert outcome.result == "done"
    assert control.monitor.test_results[0].name == "t"


def test_file_stream_invalid_json(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"kind": "tool_call", "name": "x"}\nnot json\n',
        encoding="utf-8",
    )
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "invalid json" in outcome.error.lower()
    assert "Line 2" in outcome.error


def test_file_stream_unknown_kind(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text('{"kind": "magic", "spell": "foo"}\n', encoding="utf-8")
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "unknown event kind" in outcome.error.lower()
    assert "magic" in outcome.error


def test_file_stream_missing_required_field(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text('{"kind": "command"}\n', encoding="utf-8")
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "'command' must be a string" in outcome.error


def test_file_stream_extra_fields_are_ignored(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"kind": "discovery", "text": "clue", "extra": "foo"}\n'
        '{"kind": "status", "status": "success", "result": "done"}\n',
        encoding="utf-8",
    )
    adapter = FileStreamPiAdapter(log)
    control = _make_control()

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert outcome.result == "done"
    assert control.monitor.discoveries == ["clue"]


# ---------------------------------------------------------------------------
# SubprocessPiAdapter adversarial cases
# ---------------------------------------------------------------------------


def test_subprocess_exit_zero_with_no_events(tmp_path: Path) -> None:
    script = _write_script(tmp_path, "noop.py", "import sys\nsys.exit(0)\n")
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert outcome.result == {"message": "completed"}


def test_subprocess_exit_one_with_stderr(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "fail.py",
        'import sys\nsys.stderr.write("boom\\n")\nsys.exit(1)\n',
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "boom" in outcome.error


def test_subprocess_stdout_not_json(tmp_path: Path) -> None:
    script = _write_script(tmp_path, "chat.py", "print('hello world')\n")
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "Invalid JSON from subprocess" in outcome.error
    assert "line 1" in outcome.error.lower()


def test_subprocess_binary_output(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "binary.py",
        "import sys\nsys.stdout.buffer.write(b\"\\x80\\x81\")\n",
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "binary" in outcome.error.lower()


def test_subprocess_hung_process_times_out(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "sleep.py",
        "import time\ntime.sleep(10)\n",
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=0.1,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=0.1, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    start = time.monotonic()
    outcome = adapter.run(control)
    elapsed = time.monotonic() - start

    assert outcome.status == TrajectoryStatus.ERROR
    assert "timed out" in outcome.error.lower()
    assert elapsed < 1.0


def test_subprocess_partial_line_and_exits(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "partial.py",
        'import sys\n'
        'sys.stdout.write(\'{"kind": "status", "status": "succ\')\n',
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=1.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.ERROR
    assert "Invalid JSON from subprocess" in outcome.error


def test_subprocess_ten_thousand_events(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path,
        "many.py",
        "import json\n"
        "for i in range(10000):\n"
        "    print(json.dumps({'kind': 'model_call', 'tokens': 1}))\n"
        "print(json.dumps({'kind': 'status', 'status': 'success', 'result': 'done'}))\n",
    )
    adapter = SubprocessPiAdapter(
        [sys.executable, str(script)],
        timeout_seconds=30.0,
        heartbeat_interval=0.01,
    )
    control = _make_control(timeout_seconds=30.0, heartbeat_interval=0.01)
    adapter.reset(_make_context(tmp_path))

    outcome = adapter.run(control)

    assert outcome.status == TrajectoryStatus.SUCCESS
    assert outcome.result == "done"
    assert control.monitor.model_calls == 10000
