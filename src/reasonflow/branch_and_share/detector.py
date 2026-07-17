"""Stagnation detection for branch_and_share."""

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .config import StagnationConfig
from .monitor import TrajectoryMonitor
from .results import StagnationReport


def _normalize(command: str) -> str:
    return command.strip().lower()


class StagnationDetector:
    """Detect when a Pi coding-agent trajectory is stuck."""

    def __init__(self, config: StagnationConfig) -> None:
        self.config = config

    def check(self, monitor: TrajectoryMonitor) -> Optional[StagnationReport]:
        """Analyze monitor history and return a report if stuck."""
        signals: List[str] = []
        window = self.config.window

        recent_commands = monitor.commands[-window:]
        if len(recent_commands) >= self.config.repeat_threshold:
            counts = Counter(_normalize(c["command"]) for c in recent_commands)
            if any(v >= self.config.repeat_threshold for v in counts.values()):
                signals.append("repeated_command")

        recent_reads = monitor.file_reads[-window:]
        if len(recent_reads) >= self.config.file_read_threshold:
            counts = Counter(r["path"] for r in recent_reads)
            if any(v >= self.config.file_read_threshold for v in counts.values()):
                signals.append("repeated_file_read")

        recent_failures = [t for t in monitor.tool_calls[-window:] if t.failed]
        if len(recent_failures) >= self.config.tool_failure_threshold:
            counts = Counter(t.name for t in recent_failures)
            if any(v >= self.config.tool_failure_threshold for v in counts.values()):
                signals.append("repeated_tool_failure")

        if self._no_test_improvement(monitor):
            signals.append("no_test_improvement")

        if self._code_churn(monitor):
            signals.append("code_churn")

        if monitor.total_tokens >= self.config.token_limit * self.config.token_warn_fraction:
            signals.append("approaching_token_limit")

        if not signals:
            return None

        confidence = min(1.0, 0.3 + 0.15 * len(signals))
        summary = "Stagnation signals: " + ", ".join(signals)
        return StagnationReport(signals=signals, summary=summary, confidence=confidence)

    def _no_test_improvement(self, monitor: TrajectoryMonitor) -> bool:
        recent = monitor.test_results[-self.config.test_window :]
        if len(recent) < self.config.test_window:
            return False

        seen: Dict[str, bool] = {}
        passing_counts: List[int] = []
        for t in recent:
            seen[t.name] = t.passed
            passing_counts.append(sum(1 for v in seen.values() if v))

        baseline = passing_counts[0]
        increased = any(c > baseline for c in passing_counts[1:])
        has_failures = any(not t.passed for t in recent)
        return not increased and has_failures

    def _code_churn(self, monitor: TrajectoryMonitor) -> bool:
        changes_by_file = defaultdict(list)
        for fc in monitor.file_changes:
            changes_by_file[fc.path].append(fc)

        for changes in changes_by_file.values():
            if len(changes) >= self.config.churn_threshold:
                return True
            for i in range(1, len(changes)):
                if changes[i].old_hash == changes[i - 1].new_hash:
                    return True
        return False
