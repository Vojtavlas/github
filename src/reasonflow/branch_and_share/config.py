"""Configuration dataclasses for branch_and_share."""

from dataclasses import dataclass, field


@dataclass
class StagnationConfig:
    """Thresholds for detecting a stuck trajectory."""

    repeat_threshold: int = 3
    file_read_threshold: int = 3
    test_window: int = 5
    tool_failure_threshold: int = 2
    churn_threshold: int = 3
    token_limit: int = 100000
    token_warn_fraction: float = 0.85
    window: int = 10

    def __post_init__(self) -> None:
        if self.repeat_threshold <= 0:
            raise ValueError("repeat_threshold must be > 0")
        if self.file_read_threshold <= 0:
            raise ValueError("file_read_threshold must be > 0")
        if self.test_window <= 0:
            raise ValueError("test_window must be > 0")
        if self.tool_failure_threshold <= 0:
            raise ValueError("tool_failure_threshold must be > 0")
        if self.churn_threshold <= 0:
            raise ValueError("churn_threshold must be > 0")
        if self.token_limit <= 0:
            raise ValueError("token_limit must be > 0")
        if not 0 <= self.token_warn_fraction <= 1:
            raise ValueError("token_warn_fraction must be in [0, 1]")
        if self.window <= 0:
            raise ValueError("window must be > 0")


@dataclass
class BranchAndShareConfig:
    """Top-level configuration for the branching and sharing engine."""

    max_branches: int = 3
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    max_history: int = 1000
    base_branch: str = "main"
    worktrees_dir: str = ".worktrees"
    use_git_worktrees: bool = True
    reuse_checkpoints: bool = True
    timeout_seconds: float = 60.0
    heartbeat_interval: float = 0.05

    def __post_init__(self) -> None:
        if self.max_history < 0:
            raise ValueError("max_history must be >= 0")
        if self.max_branches < 0:
            raise ValueError("max_branches must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be > 0")
