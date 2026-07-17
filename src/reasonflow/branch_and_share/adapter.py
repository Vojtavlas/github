"""Adapters for integrating a Pi coding agent with branch_and_share."""

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from .control import TrajectoryControl
from .results import BranchContext, StagnationReport, TrajectoryOutcome, TrajectoryStatus


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


def _apply_event(event: Dict[str, Any], control: TrajectoryControl) -> Optional[TrajectoryOutcome]:
    """Replay a single JSON event into ``control``.

    Returns an outcome only when the event is an explicit ``status`` event.
    """
    kind = event.get("kind")
    if kind == "tool_call":
        control.record_tool_call(
            name=event["name"],
            args=event.get("args", {}),
            result=event.get("result"),
            failed=event.get("failed", False),
            tokens=event.get("tokens", 0),
        )
    elif kind == "file_read":
        control.record_file_read(
            path=event["path"],
            symbols=event.get("symbols"),
        )
    elif kind == "command":
        control.record_command(
            command=event["command"],
            output=event.get("output", ""),
            exit_code=event.get("exit_code", 0),
        )
    elif kind == "test":
        control.record_test_result(
            name=event["name"],
            passed=event.get("passed", True),
            output=event.get("output", ""),
        )
    elif kind == "file_change":
        control.record_file_change(
            path=event["path"],
            change_type=event.get("change_type", "modified"),
            old_hash=event.get("old_hash"),
            new_hash=event.get("new_hash"),
        )
    elif kind == "model_call":
        control.record_model_call(tokens=event.get("tokens", 0))
    elif kind == "token_usage":
        control.record_token_usage(event.get("n", 0))
    elif kind == "hypothesis":
        control.record_hypothesis(
            description=event["description"],
            action=event["action"],
            expected_test_change=event["expected"],
            observed=event.get("observed", ""),
            failed=event.get("failed", True),
        )
    elif kind == "discovery":
        control.record_discovery(event["text"])
    elif kind == "status":
        status = event.get("status")
        if status == "success":
            return TrajectoryOutcome(
                status=TrajectoryStatus.SUCCESS,
                result=event.get("result"),
            )
        if status == "error":
            return TrajectoryOutcome(
                status=TrajectoryStatus.ERROR,
                error=event.get("error", ""),
            )
        if status == "stagnation":
            report = control.check_stagnation()
            if report is None:
                report = StagnationReport(
                    signals=["status"],
                    summary="External status reported stagnation",
                    confidence=1.0,
                )
            return TrajectoryOutcome(status=TrajectoryStatus.STAGNATION, report=report)
        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result=event.get("result"),
        )
    else:
        raise ValueError(f"Unknown event kind: {kind}")
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
        for raw_line in self._iter_lines(stream):
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            outcome = _apply_event(event, control)
            if outcome is not None:
                return outcome
            stale = _check_stagnation(control)
            if stale is not None:
                return stale
        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result={"message": "completed"},
        )


class SubprocessPiAdapter(PiAdapter):
    """Run an external Pi-agent process and replay its JSON event stream.

    The subprocess is launched in the branch worktree with ``BRANCH_ID`` and
    ``BRANCH_CONTEXT_PATH`` environment variables. Its ``stdout`` is consumed
    line-by-line as JSON events.
    """

    def __init__(
        self,
        command: List[str],
        context_path: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__()
        self.command = command
        self.context_path = context_path
        self.env = env or {}
        self.poll_interval = poll_interval

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

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        if self.context is None:
            raise RuntimeError("SubprocessPiAdapter.run() called before reset()")
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
        if process.stdout is not None:
            for raw_line in iter(process.stdout.readline, ""):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    outcome = TrajectoryOutcome(
                        status=TrajectoryStatus.ERROR,
                        error=f"Invalid JSON from subprocess: {exc}",
                    )
                    break
                outcome = _apply_event(event, control)
                if outcome is not None:
                    break
                stale = _check_stagnation(control)
                if stale is not None:
                    outcome = stale
                    break

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
        elif kind == "tokens":
            control.record_token_usage(step["n"])
        elif kind == "model_call":
            control.record_model_call(tokens=step.get("tokens", 0))
        else:
            raise ValueError(f"Unknown step kind: {kind}")
