"""Metrics tracking for branch_and_share."""

import time
from typing import Optional

from .results import BranchMetrics


class MetricsTracker:
    """Track tokens, calls, tests, and wall-clock time for one branch."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._stop: Optional[float] = None
        self.total_tokens = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.duplicated = 0
        self.tests_passing = 0
        self.tests_failing = 0

    def start(self) -> None:
        """Mark the start of a branch."""
        self._start = time.time()

    def stop(self) -> None:
        """Mark the end of a branch."""
        self._stop = time.time()

    def add_tokens(self, n: int) -> None:
        self.total_tokens += n

    def add_model_call(self) -> None:
        self.model_calls += 1

    def add_tool_call(self) -> None:
        self.tool_calls += 1

    def add_duplicated(self, n: int = 1) -> None:
        self.duplicated += n

    def set_test_progress(self, passing: int, failing: int) -> None:
        self.tests_passing = passing
        self.tests_failing = failing

    def snapshot(self, branch_count: int = 0, final_success: bool = False) -> BranchMetrics:
        """Return a frozen metrics snapshot."""
        elapsed = 0.0
        if self._start is not None and self._stop is not None:
            elapsed = (self._stop - self._start) * 1000.0
        return BranchMetrics(
            total_tokens=self.total_tokens,
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            duplicated_work=self.duplicated,
            branch_count=branch_count,
            tests_passing=self.tests_passing,
            tests_failing=self.tests_failing,
            final_success=final_success,
            wall_clock_ms=elapsed,
        )
