"""Experience packet builder for branch_and_share."""

from typing import Dict, List, Optional

from .monitor import TrajectoryMonitor
from .results import (
    BranchContext,
    BranchMetrics,
    ExperiencePacket,
    StagnationReport,
    TestResult,
)


class ExperiencePacketBuilder:
    """Build a portable experience packet grounded in git, logs, and tests."""

    def __init__(
        self,
        monitor: TrajectoryMonitor,
        branch_context: BranchContext,
        metrics: BranchMetrics,
    ) -> None:
        self.monitor = monitor
        self.branch_context = branch_context
        self.metrics = metrics

    def build(self, report: Optional[StagnationReport] = None) -> ExperiencePacket:
        git_state = self.monitor.current_git_state()
        latest_tests = self._latest_tests()

        passing_names = [name for name, t in latest_tests.items() if t.passed]
        failing_names = [name for name, t in latest_tests.items() if not t.passed]

        commands = self.monitor.command_log()
        tests = [f"{t.name}: {'PASS' if t.passed else 'FAIL'}" for t in latest_tests.values()]
        commands_and_tests = commands + tests
        if git_state.modified_files:
            commands_and_tests.append(
                f"Modified files: {', '.join(git_state.modified_files)}"
            )

        evidence: List[str] = []
        for t in latest_tests.values():
            if not t.passed:
                evidence.append(f"Failing test: {t.name}\n{t.output}")
        for tc in self.monitor.tool_calls:
            if tc.failed:
                evidence.append(f"Failed tool: {tc.name} with args {tc.args}")

        recommendations = self._recommendations(report, failing_names)

        return ExperiencePacket(
            files_and_symbols_inspected=self.monitor.get_files_and_symbols_inspected(),
            commands_and_tests_run=commands_and_tests,
            modified_files_and_diff=git_state.diff,
            current_passing_tests=passing_names,
            current_failing_tests=failing_names,
            hypotheses_attempted=self.monitor.get_hypotheses(),
            evidence_of_failure=evidence,
            useful_discoveries=self.monitor.get_useful_discoveries(),
            recommended_next_actions=recommendations,
            metrics=self.metrics,
        )

    def _latest_tests(self) -> Dict[str, TestResult]:
        latest: Dict[str, TestResult] = {}
        for t in self.monitor.test_results:
            latest[t.name] = t
        return latest

    def _recommendations(
        self, report: Optional[StagnationReport], failing_names: List[str]
    ) -> List[str]:
        recs: List[str] = []
        signals = report.signals if report else []

        if failing_names:
            recs.append(f"Investigate failing tests: {failing_names[:3]}")
        if "code_churn" in signals:
            recs.append(
                "Pause editing churning files and run tests before further changes."
            )
        if "repeated_command" in signals:
            recs.append(
                "Avoid repeating the same command; inspect output and change strategy."
            )
        if "repeated_file_read" in signals:
            recs.append(
                "Avoid re-reading the same files; consolidate knowledge before acting."
            )
        if "repeated_tool_failure" in signals:
            recs.append("Check tool arguments and environment before retrying.")
        if "approaching_token_limit" in signals:
            recs.append("Reduce scope or switch to a shorter plan.")
        if "no_test_improvement" in signals:
            recs.append(
                "Run tests after each change and verify the failure signal changes."
            )
        if not recs:
            recs.append("Summarize current state and pick a focused next hypothesis.")

        return recs
