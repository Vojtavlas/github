"""Trajectory monitor for the branch-and-share layer."""

import os
import subprocess
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .results import (
    FileChange,
    GitState,
    HypothesisAttempt,
    InspectRecord,
    TestResult,
    ToolCall,
)


def _hashable(value: Any) -> Any:
    """Return a hashable representation of a value for duplicate counting."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _tool_key(call: ToolCall) -> Tuple[str, Any]:
    return (call.name, _hashable(call.args))


class TrajectoryMonitor:
    """Record and query a single Pi coding-agent trajectory."""

    def __init__(self, git_repo_root: Optional[str] = None) -> None:
        self.git_repo_root = git_repo_root or os.getcwd()
        self.tool_calls: List[ToolCall] = []
        self.file_reads: List[Dict[str, Any]] = []
        self.commands: List[Dict[str, Any]] = []
        self.test_results: List[TestResult] = []
        self.file_changes: List[FileChange] = []
        self.hypotheses: List[HypothesisAttempt] = []
        self.discoveries: List[str] = []
        self.inspected: List[InspectRecord] = []
        self.total_tokens = 0
        self.model_calls = 0

    def record_tool_call(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        failed: bool = False,
        tokens: int = 0,
    ) -> ToolCall:
        """Record a tool invocation and token cost."""
        call = ToolCall(
            name=name,
            args=args or {},
            result=result,
            failed=failed,
            timestamp=time.time(),
            tokens=tokens,
        )
        self.tool_calls.append(call)
        self.total_tokens += tokens
        return call

    def record_file_read(self, path: str, symbols: Optional[List[str]] = None) -> None:
        """Record that a file (and optionally specific symbols) was inspected."""
        self.file_reads.append(
            {"path": path, "symbols": symbols, "timestamp": time.time()}
        )
        self.inspected.append(
            InspectRecord(path=path, symbols=symbols or [], timestamp=time.time())
        )

    def record_command(
        self, command: str, output: str = "", exit_code: int = 0
    ) -> None:
        """Record a shell command that was run."""
        self.commands.append(
            {
                "command": command,
                "output": output,
                "exit_code": exit_code,
                "timestamp": time.time(),
            }
        )

    def record_test_result(self, name: str, passed: bool, output: str = "") -> None:
        """Record a test execution."""
        self.test_results.append(
            TestResult(name=name, passed=passed, output=output, timestamp=time.time())
        )

    def record_model_call(self, tokens: int = 0) -> None:
        """Record a model call and associated token usage."""
        self.model_calls += 1
        self.total_tokens += tokens

    def record_token_usage(self, n: int) -> None:
        """Record extra token usage."""
        self.total_tokens += n

    def record_file_change(
        self,
        path: str,
        change_type: str = "modified",
        old_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
    ) -> None:
        """Record that a file was modified."""
        self.file_changes.append(
            FileChange(
                path=path,
                change_type=change_type,
                old_hash=old_hash,
                new_hash=new_hash,
                timestamp=time.time(),
            )
        )

    def record_hypothesis(
        self,
        description: str,
        action: str,
        expected_test_change: str,
        observed: str = "",
        failed: bool = True,
    ) -> None:
        """Record a hypothesis the agent attempted and its outcome."""
        self.hypotheses.append(
            HypothesisAttempt(
                description=description,
                action=action,
                expected_test_change=expected_test_change,
                observed=observed,
                failed=failed,
                timestamp=time.time(),
            )
        )

    def record_discovery(self, text: str) -> None:
        """Record a useful discovery, deduplicated."""
        if text not in self.discoveries:
            self.discoveries.append(text)

    def current_git_state(self) -> GitState:
        """Return the current git branch, modified files, and diff.

        This is grounded in the real repository state, not the model summary.
        """
        if not self._is_git_repo():
            return GitState()
        try:
            branch = self._git("branch", "--show-current").strip()
            status = self._git("status", "--porcelain")
            diff = self._git("diff")
            modified = [
                line[3:].strip() for line in status.splitlines() if len(line) > 3
            ]
            return GitState(branch=branch, modified_files=modified, diff=diff)
        except (subprocess.CalledProcessError, OSError):
            return GitState()

    def _is_git_repo(self) -> bool:
        if not os.path.isdir(self.git_repo_root):
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.git_repo_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.git_repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def test_progress(self) -> Tuple[int, int, int]:
        """Return (passing, failing, total) for the latest unique test results."""
        latest: Dict[str, TestResult] = {}
        for t in self.test_results:
            latest[t.name] = t
        passing = sum(1 for t in latest.values() if t.passed)
        failing = sum(1 for t in latest.values() if not t.passed)
        return passing, failing, len(latest)

    def duplicated_work(self, window: int = 20) -> int:
        """Count repeated tool calls and file reads in the recent window."""
        recent_tools = self.tool_calls[-window:]
        recent_reads = self.file_reads[-window:]
        tool_counts = Counter(_tool_key(c) for c in recent_tools)
        read_counts = Counter(r["path"] for r in recent_reads)
        duplicates = 0
        for count in tool_counts.values():
            if count > 1:
                duplicates += count - 1
        for count in read_counts.values():
            if count > 1:
                duplicates += count - 1
        return duplicates

    def command_log(self, limit: int = 20) -> List[str]:
        """Return recent commands as human-readable strings."""
        return [
            f"{c['command']} (exit {c['exit_code']})" for c in self.commands[-limit:]
        ]

    def test_output(self) -> str:
        """Return concatenated output from recent failing tests."""
        failing = [t for t in self.test_results if not t.passed]
        return "\n\n".join(t.output for t in failing[-5:] if t.output)

    def get_files_and_symbols_inspected(self) -> List[InspectRecord]:
        """Return the latest inspect record per file path."""
        by_path: Dict[str, InspectRecord] = {}
        for record in self.inspected:
            by_path[record.path] = record
        return list(by_path.values())

    def get_hypotheses(self) -> List[HypothesisAttempt]:
        """Return all recorded hypotheses."""
        return list(self.hypotheses)

    def get_useful_discoveries(self) -> List[str]:
        """Return deduplicated useful discoveries."""
        return list(self.discoveries)
