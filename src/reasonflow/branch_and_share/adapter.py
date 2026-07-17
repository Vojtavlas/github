"""Adapters for integrating a Pi coding agent with branch_and_share."""

import json
import os
import queue
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from .control import TrajectoryControl
from .protocol import (
    CommandEvent,
    DiscoveryEvent,
    Event,
    FileChangeEvent,
    FileReadEvent,
    HypothesisEvent,
    MalformedEventError,
    ModelCallEvent,
    StatusEvent,
    TestEvent,
    TokenUsageEvent,
    ToolCallEvent,
    _validate_event,
)
from .results import BranchContext, StagnationReport, TrajectoryOutcome, TrajectoryStatus
from .stream import EventStreamReader


class TrajectoryRunner(ABC):
    """Abstract runner for a single agent trajectory."""

    @abstractmethod
    def reset(self, context: BranchContext) -> None:
        """Prepare to run a new branch."""

    @abstractmethod
    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        """Execute the trajectory, reporting events through ``control``."""


class PiAdapter(TrajectoryRunner):
    """Plugin seam for a Pi coding agent.

    Subclass this and implement :meth:`run` to integrate the real Pi agent.
    The Pi agent should call the provided ``control`` methods before/after
    tool calls and check :meth:`TrajectoryControl.check_stagnation` to decide
    when to stop and let the engine branch.
    """

    def __init__(self) -> None:
        self.context: Optional[BranchContext] = None

    def reset(self, context: BranchContext) -> None:
        self.context = context

    @abstractmethod
    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        ...


def _apply_event(event: Event, control: TrajectoryControl) -> Optional[TrajectoryOutcome]:
    """Replay a validated ``Event`` into ``control``.

    Returns an outcome only when the event is an explicit ``status`` event.
    """
    if isinstance(event, ToolCallEvent):
        control.record_tool_call(
            name=event.name,
            args=event.args or {},
            result=event.result,
            failed=event.failed,
            tokens=event.tokens,
        )
    elif isinstance(event, FileReadEvent):
        control.record_file_read(
            path=event.path,
            symbols=event.symbols,
        )
    elif isinstance(event, CommandEvent):
        control.record_command(
            command=event.command,
            output=event.output,
            exit_code=event.exit_code,
        )
    elif isinstance(event, TestEvent):
        control.record_test_result(
            name=event.name,
            passed=event.passed,
            output=event.output,
        )
    elif isinstance(event, FileChangeEvent):
        control.record_file_change(
            path=event.path,
            change_type=event.change_type,
            old_hash=event.old_hash,
            new_hash=event.new_hash,
        )
    elif isinstance(event, ModelCallEvent):
        control.record_model_call(tokens=event.tokens)
    elif isinstance(event, TokenUsageEvent):
        control.record_token_usage(event.n)
    elif isinstance(event, HypothesisEvent):
        control.record_hypothesis(
            description=event.description,
            action=event.action,
            expected_test_change=event.expected_test_change,
            observed=event.observed,
            failed=event.failed,
        )
    elif isinstance(event, DiscoveryEvent):
        control.record_discovery(event.text)
    elif isinstance(event, StatusEvent):
        if event.status == "success":
            return TrajectoryOutcome(
                status=TrajectoryStatus.SUCCESS,
                result=event.result,
            )
        if event.status == "error":
            return TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error=event.error,
            )
        if event.status == "stagnation":
            report = control.check_stagnation()
            if report is None:
                report = StagnationReport(
                    signals=["status"],
                    summary="External status reported stagnation",
                    confidence=1.0,
                )
            return TrajectoryOutcome(status=TrajectoryStatus.STAGNATION, report=report)
        raise MalformedEventError(
            f"Unrecognized status value: {event.status!r}", event.line_no
        )
    else:
        raise MalformedEventError(
            f"Unknown event kind: {event.kind!r}", event.line_no
        )
    return None


def _check_stagnation(control: TrajectoryControl) -> Optional[TrajectoryOutcome]:
    """Return a stagnation outcome if the detector fires."""
    report = control.check_stagnation()
    if report is not None:
        return TrajectoryOutcome(status=TrajectoryStatus.STAGNATION, report=report)
    return None


class FileStreamPiAdapter(PiAdapter):
    """Replay a Pi trajectory from a newline-delimited JSON event log.

    ``source`` may be a file path or an open text stream. Each line must be a
    JSON object with a ``kind`` field. The adapter replays events into the
    provided :class:`TrajectoryControl` and stops at the first ``status``
    event or at EOF. Stagnation is checked after every event.
    """

    def __init__(
        self,
        source: Union[str, Path, TextIO],
        poll_interval: float = 0.1,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _iter_lines(self, stream: TextIO) -> Any:
        """Yield lines from ``stream`` until EOF or timeout."""
        deadline = None if self.timeout is None else time.time() + self.timeout
        while True:
            line = stream.readline()
            if line:
                yield line
                continue
            if deadline is not None and time.time() >= deadline:
                break
            # Wait briefly for more data in case the stream is still being
            # written (e.g. a live log tail).
            if self.timeout is None and not line:
                break
            time.sleep(self.poll_interval)

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        if isinstance(self.source, (str, Path)):
            with Path(self.source).open("r", encoding="utf-8") as stream:
                return self._run_stream(stream, control)
        return self._run_stream(self.source, control)

    def _run_stream(self, stream: TextIO, control: TrajectoryControl) -> TrajectoryOutcome:
        reader = EventStreamReader(self._iter_lines(stream))
        try:
            for event in reader:
                outcome = _apply_event(event, control)
                if outcome is not None:
                    return outcome
                stale = _check_stagnation(control)
                if stale is not None:
                    return stale
        except MalformedEventError as exc:
            return TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error=str(exc),
            )
        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result={"message": "completed"},
        )


class SubprocessPiAdapter(PiAdapter):
    """Run an external Pi-agent process and replay its JSON event stream.

    The subprocess is launched in the branch worktree with ``BRANCH_ID`` and
    ``BRANCH_CONTEXT_PATH`` environment variables. Its ``stdout`` is consumed
    line-by-line in a dedicated reader thread and placed on a queue. The main
    thread applies events with a ``heartbeat_interval`` poll and an overall
    ``timeout_seconds`` guard, checking for stagnation between events.
    """

    def __init__(
        self,
        command: List[str],
        context_path: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        heartbeat_interval: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.command = command
        self.context_path = context_path
        self.env = env or {}
        self.timeout_seconds = timeout_seconds
        self.heartbeat_interval = heartbeat_interval

    def reset(self, context: BranchContext) -> None:
        super().reset(context)
        if self.context_path is None and context is not None:
            self.context_path = str(Path(context.worktree_path) / ".branch_context.json")

    def _build_env(self) -> Dict[str, str]:
        environment = dict(os.environ)
        environment.update(self.env)
        if self.context_path is not None:
            environment["BRANCH_CONTEXT_PATH"] = self.context_path
        if self.context is not None:
            environment["BRANCH_ID"] = str(self.context.branch_id)
            environment["BRANCH_WORKTREE"] = str(self.context.worktree_path)
        return environment

    def _stream_lines(self, stdout: TextIO, q: "queue.Queue[Optional[str]]") -> None:
        """Read lines from ``stdout`` and push them onto ``q``."""
        try:
            for line in iter(stdout.readline, ""):
                q.put(line)
        finally:
            q.put(None)

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        if self.context is None:
            raise RuntimeError("SubprocessPiAdapter.run() called before reset()")

        timeout = (
            self.timeout_seconds
            if self.timeout_seconds is not None
            else control.config.timeout_seconds
        )
        heartbeat = (
            self.heartbeat_interval
            if self.heartbeat_interval is not None
            else control.config.heartbeat_interval
        )

        process = subprocess.Popen(
            self.command,
            cwd=self.context.worktree_path,
            env=self._build_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        outcome: Optional[TrajectoryOutcome] = None
        reader_thread: Optional[threading.Thread] = None

        try:
            if process.stdout is None:
                raise RuntimeError("Subprocess stdout is not available")

            q: "queue.Queue[Optional[str]]" = queue.Queue()
            reader_thread = threading.Thread(
                target=self._stream_lines,
                args=(process.stdout, q),
                daemon=True,
            )
            reader_thread.start()

            deadline = time.time() + timeout
            line_no = 0
            while True:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    outcome = TrajectoryOutcome(
                        status=TrajectoryStatus.ERROR,
                        error="Subprocess timed out",
                    )
                    break

                get_timeout = min(heartbeat, remaining)
                try:
                    line = q.get(timeout=get_timeout)
                except queue.Empty:
                    stale = _check_stagnation(control)
                    if stale is not None:
                        outcome = stale
                        break
                    continue

                if line is None:
                    break

                line_no += 1
                text = line.rstrip("\r\n")
                if not text.strip():
                    continue

                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    outcome = TrajectoryOutcome(
                        status=TrajectoryStatus.ERROR,
                        error=f"Invalid JSON from subprocess: {exc}",
                    )
                    break

                try:
                    event = _validate_event(data, line_no)
                except MalformedEventError as exc:
                    outcome = TrajectoryOutcome(
                        status=TrajectoryStatus.ERROR,
                        error=str(exc),
                    )
                    break

                outcome = _apply_event(event, control)
                if outcome is not None:
                    break

                stale = _check_stagnation(control)
                if stale is not None:
                    outcome = stale
                    break
        finally:
            if outcome is not None and outcome.status != TrajectoryStatus.SUCCESS:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            else:
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            if reader_thread is not None and reader_thread.is_alive():
                # Closing stdout unblocks the reader thread's readline call.
                try:
                    process.stdout.close()
                except OSError:
                    pass
                reader_thread.join(timeout=2.0)

        if process.returncode != 0 and outcome is None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            outcome = TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error=stderr or f"Subprocess exited with code {process.returncode}",
            )

        return outcome or TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result={"message": "completed"},
        )


class TailPiAdapter(PiAdapter):
    """Tail a growing newline-delimited JSON log file in real time.

    The adapter opens ``path``, starts at the current end-of-file, and polls
    for newly appended lines every ``heartbeat_interval``. It replays each
    line through :class:`TrajectoryControl` and returns on the first ``status``
    event. If the file shrinks (log rotation or truncation), it resets to the
    new end-of-file rather than re-reading from the beginning.
    """

    def __init__(
        self,
        path: Union[str, Path],
        timeout_seconds: Optional[float] = None,
        heartbeat_interval: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.heartbeat_interval = heartbeat_interval

    def _resolve_timeouts(self, control: TrajectoryControl) -> tuple[float, float]:
        timeout = (
            self.timeout_seconds
            if self.timeout_seconds is not None
            else control.config.timeout_seconds
        )
        heartbeat = (
            self.heartbeat_interval
            if self.heartbeat_interval is not None
            else control.config.heartbeat_interval
        )
        return timeout, heartbeat

    def _iter_tail_lines(
        self,
        stream: TextIO,
        heartbeat: float,
        deadline: float,
    ) -> Any:
        """Yield new lines from ``stream`` until timeout or EOF sentinel."""
        while time.time() < deadline:
            line = stream.readline()
            if line:
                yield line
                continue

            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                size = 0

            pos = stream.tell()
            if size < pos:
                # Log was truncated/rotated; jump to the new end.
                stream.seek(size)
            elif size > pos:
                # New data is already available; readline will pick it up.
                continue
            else:
                time.sleep(heartbeat)

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        timeout, heartbeat = self._resolve_timeouts(control)
        deadline = time.time() + timeout

        try:
            with self.path.open("r", encoding="utf-8") as stream:
                stream.seek(0, 2)
                reader = EventStreamReader(
                    self._iter_tail_lines(stream, heartbeat, deadline)
                )
                try:
                    for event in reader:
                        outcome = _apply_event(event, control)
                        if outcome is not None:
                            return outcome
                        stale = _check_stagnation(control)
                        if stale is not None:
                            return stale
                except MalformedEventError as exc:
                    return TrajectoryOutcome(
                        status=TrajectoryStatus.ERROR,
                        error=str(exc),
                    )
        except FileNotFoundError as exc:
            return TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error=f"Log file not found: {self.path} ({exc})",
            )

        if time.time() >= deadline:
            return TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error="Tail timed out",
            )

        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result={"message": "completed"},
        )


class MockPiAdapter(TrajectoryRunner):
    """Scripted adapter used for end-to-end tests."""

    def __init__(
        self,
        scenario: Optional[List[Dict[str, Any]]] = None,
        scenarios: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        default_scenario: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.scenario = scenario or []
        self.scenarios = scenarios or {}
        self.default_scenario = default_scenario or []
        self._current: List[Dict[str, Any]] = []
        self._index = 0
        self.context: Optional[BranchContext] = None

    def reset(self, context: BranchContext) -> None:
        self.context = context
        if self.scenarios:
            self._current = self.scenarios.get(
                context.branch_id, self.default_scenario
            )
        else:
            self._current = self.scenario
        if not self._current and self.scenario:
            self._current = self.scenario
        self._index = 0

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        while self._index < len(self._current):
            step = self._current[self._index]
            self._index += 1
            self._apply_step(step, control)
            report = control.check_stagnation()
            if report is not None:
                return TrajectoryOutcome(
                    status=TrajectoryStatus.STAGNATION, report=report
                )
        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS, result={"message": "completed"}
        )

    def _apply_step(self, step: Dict[str, Any], control: TrajectoryControl) -> None:
        kind = step.get("kind")
        if kind == "tool_call":
            control.record_tool_call(
                name=step["name"],
                args=step.get("args", {}),
                result=step.get("result"),
                failed=step.get("failed", False),
                tokens=step.get("tokens", 0),
            )
        elif kind == "file_read":
            control.record_file_read(
                path=step["path"],
                symbols=step.get("symbols"),
            )
        elif kind == "command":
            control.record_command(
                command=step["command"],
                output=step.get("output", ""),
                exit_code=step.get("exit_code", 0),
            )
        elif kind == "test":
            control.record_test_result(
                name=step["name"],
                passed=step.get("passed", True),
                output=step.get("output", ""),
            )
        elif kind == "file_change":
            control.record_file_change(
                path=step["path"],
                change_type=step.get("change_type", "modified"),
                old_hash=step.get("old_hash"),
                new_hash=step.get("new_hash"),
            )
        elif kind == "hypothesis":
            control.record_hypothesis(
                description=step["description"],
                action=step["action"],
                expected_test_change=step["expected"],
                observed=step.get("observed", ""),
                failed=step.get("failed", True),
            )
        elif kind == "discovery":
            control.record_discovery(step["text"])
        elif kind in ("token_usage", "tokens"):
            control.record_token_usage(step["n"])
        elif kind == "model_call":
            control.record_model_call(tokens=step.get("tokens", 0))
        else:
            raise ValueError(f"Unknown step kind: {kind}")
