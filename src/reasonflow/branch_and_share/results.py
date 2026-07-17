"""Dataclasses for the failure-aware branching and sharing layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TrajectoryStatus(str, Enum):
    """Final status of a single trajectory."""

    SUCCESS = "success"
    STAGNATION = "stagnation"
    ERROR = "error"


@dataclass
class ToolCall:
    """A recorded tool invocation."""

    name: str
    args: Dict[str, Any]
    result: Any
    failed: bool = False
    timestamp: float = 0.0
    tokens: int = 0


@dataclass
class TestResult:
    """A recorded test execution."""

    name: str
    passed: bool
    output: str = ""
    timestamp: float = 0.0


@dataclass
class FileChange:
    """A recorded change to a file."""

    path: str
    change_type: str = "modified"
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class HypothesisAttempt:
    """A hypothesis the agent attempted and how it fared."""

    description: str
    action: str
    expected_test_change: str
    observed: str = ""
    failed: bool = True
    timestamp: float = 0.0


@dataclass
class InspectRecord:
    """A file read with the symbols inspected."""

    path: str
    symbols: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class GitState:
    """Snapshot of the current git working tree."""

    branch: str = ""
    modified_files: List[str] = field(default_factory=list)
    diff: str = ""


@dataclass
class BranchContext:
    """Context for a single reasoning branch."""

    branch_id: int
    worktree_path: str
    start_ref: str
    start_commit: str
    summary: str = ""
    base_branch: str = "main"
    parent_branch_id: Optional[int] = None
    final_packet: Optional["ExperiencePacket"] = None


@dataclass
class StagnationReport:
    """Report produced when a trajectory appears stuck."""

    signals: List[str]
    summary: str
    confidence: float = 0.0


@dataclass
class ExperiencePacket:
    """Portable summary of a branch to share with future branches."""

    files_and_symbols_inspected: List[InspectRecord]
    commands_and_tests_run: List[str]
    modified_files_and_diff: str
    current_passing_tests: List[str]
    current_failing_tests: List[str]
    hypotheses_attempted: List[HypothesisAttempt]
    evidence_of_failure: List[str]
    useful_discoveries: List[str]
    recommended_next_actions: List[str]
    metrics: "BranchMetrics"


@dataclass
class TrajectoryOutcome:
    """Result of running a single trajectory."""

    status: TrajectoryStatus
    result: Optional[Any] = None
    report: Optional[StagnationReport] = None
    error: Optional[str] = None


@dataclass
class BranchMetrics:
    """Aggregated metrics across branches."""

    total_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    duplicated_work: int = 0
    branch_count: int = 0
    tests_passing: int = 0
    tests_failing: int = 0
    final_success: bool = False
    wall_clock_ms: float = 0.0


@dataclass
class ShareResult:
    """Final result of the BranchAndShare engine."""

    best_branch_id: Optional[int]
    branches: List[BranchContext]
    success: bool
    metrics: BranchMetrics
    final_packet: Optional[ExperiencePacket] = None
@dataclass
class BranchSessionReport:
    """Summary report for a branch-and-share session."""

    branches: List[BranchContext]
    total_time_ms: float
    pass_count: int
    fail_count: int
    final_success: bool
    final_outcome: Optional[TrajectoryStatus]
    metrics: BranchMetrics
