"""Control surface passed to a trajectory runner."""

import time
from typing import Any, Dict, List, Optional

from .config import BranchAndShareConfig
from .detector import StagnationDetector
from .logging import BranchSessionLogger
from .metrics import MetricsTracker
from .monitor import TrajectoryMonitor
from .results import StagnationReport


class TrajectoryControl:
    """Interface the runner uses to report events and check stagnation."""

    def __init__(
        self,
        monitor: TrajectoryMonitor,
        detector: StagnationDetector,
        metrics: MetricsTracker,
        config: BranchAndShareConfig,
        logger: Optional[BranchSessionLogger] = None,
        branch_id: Optional[int] = None,
    ) -> None:
        self.monitor = monitor
        self.detector = detector
        self.metrics = metrics
        self.config = config
        self.logger = logger
        self.branch_id = branch_id
        self._start_time = time.time()

    def record_tool_call(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        failed: bool = False,
        tokens: int = 0,
    ) -> None:
        self.monitor.record_tool_call(name, args, result, failed, tokens)
        self.metrics.add_tool_call()
        self.metrics.add_tokens(tokens)
        self._log(
            "tool_call",
            {
                "name": name,
                "args": args or {},
                "failed": failed,
                "tokens": tokens,
            },
        )

    def record_file_read(self, path: str, symbols: Optional[List[str]] = None) -> None:
        self.monitor.record_file_read(path, symbols)
        self._log("file_read", {"path": path, "symbols": symbols or []})

    def record_command(
        self, command: str, output: str = "", exit_code: int = 0
    ) -> None:
        self.monitor.record_command(command, output, exit_code)
        self._log("command", {"command": command, "exit_code": exit_code})

    def record_test_result(self, name: str, passed: bool, output: str = "") -> None:
        self.monitor.record_test_result(name, passed, output)
        passing, failing, _ = self.monitor.test_progress()
        self.metrics.set_test_progress(passing, failing)
        self._log("test", {"name": name, "passed": passed})

    def record_model_call(self, tokens: int = 0) -> None:
        self.monitor.record_model_call(tokens)
        self.metrics.add_model_call()
        self.metrics.add_tokens(tokens)
        self._log("model_call", {"tokens": tokens})

    def record_token_usage(self, n: int) -> None:
        self.monitor.record_token_usage(n)
        self.metrics.add_tokens(n)
        self._log("token_usage", {"tokens": n})

    def record_file_change(
        self,
        path: str,
        change_type: str = "modified",
        old_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
    ) -> None:
        self.monitor.record_file_change(path, change_type, old_hash, new_hash)
        self._log(
            "file_change",
            {
                "path": path,
                "change_type": change_type,
                "old_hash": old_hash,
                "new_hash": new_hash,
            },
        )

    def record_hypothesis(
        self,
        description: str,
        action: str,
        expected_test_change: str,
        observed: str = "",
        failed: bool = True,
    ) -> None:
        self.monitor.record_hypothesis(
            description, action, expected_test_change, observed, failed
        )
        self._log(
            "hypothesis",
            {
                "description": description,
                "action": action,
                "expected_test_change": expected_test_change,
                "failed": failed,
            },
        )

    def record_discovery(self, text: str) -> None:
        self.monitor.record_discovery(text)
        self._log("discovery", {"text": text})

    def check_stagnation(self) -> Optional[StagnationReport]:
        return self.detector.check(self.monitor)

    def _elapsed_ms(self) -> float:
        start = self.metrics._start if self.metrics._start is not None else self._start_time
        return (time.time() - start) * 1000.0

    def _log(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.logger is None or self.branch_id is None:
            return
        report = self.detector.check(self.monitor)
        signals = report.signals if report is not None else []
        self.logger.log_event(
            branch_id=self.branch_id,
            kind=kind,
            elapsed_ms=self._elapsed_ms(),
            stagnation_signals=signals,
            payload=payload,
        )

    def current_git_state(self):
        return self.monitor.current_git_state()
