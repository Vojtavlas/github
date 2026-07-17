"""Control surface passed to a trajectory runner."""

from typing import Any, Dict, List, Optional

from .config import BranchAndShareConfig
from .detector import StagnationDetector
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
    ) -> None:
        self.monitor = monitor
        self.detector = detector
        self.metrics = metrics
        self.config = config

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

    def record_file_read(self, path: str, symbols: Optional[List[str]] = None) -> None:
        self.monitor.record_file_read(path, symbols)

    def record_command(
        self, command: str, output: str = "", exit_code: int = 0
    ) -> None:
        self.monitor.record_command(command, output, exit_code)

    def record_test_result(self, name: str, passed: bool, output: str = "") -> None:
        self.monitor.record_test_result(name, passed, output)
        passing, failing, _ = self.monitor.test_progress()
        self.metrics.set_test_progress(passing, failing)

    def record_model_call(self, tokens: int = 0) -> None:
        self.monitor.record_model_call(tokens)
        self.metrics.add_model_call()
        self.metrics.add_tokens(tokens)

    def record_token_usage(self, n: int) -> None:
        self.monitor.record_token_usage(n)
        self.metrics.add_tokens(n)

    def record_file_change(
        self,
        path: str,
        change_type: str = "modified",
        old_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
    ) -> None:
        self.monitor.record_file_change(path, change_type, old_hash, new_hash)

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

    def record_discovery(self, text: str) -> None:
        self.monitor.record_discovery(text)

    def check_stagnation(self) -> Optional[StagnationReport]:
        return self.detector.check(self.monitor)

    def current_git_state(self):
        return self.monitor.current_git_state()
