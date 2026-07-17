"""Failure-aware branching and sharing layer for Pi coding agents."""

from .adapter import (
    FileStreamPiAdapter,
    MockPiAdapter,
    PiAdapter,
    SubprocessPiAdapter,
    TailPiAdapter,
    TrajectoryRunner,
)
from .branch_manager import (
    BranchManager,
    BranchStartPoint,
    GitWorktreeBranchManager,
    MemoryBranchManager,
)
from .config import BranchAndShareConfig, StagnationConfig
from .control import TrajectoryControl
from .detector import StagnationDetector
from .engine import BranchAndShareEngine
from .launcher import BranchSessionLauncher
from .metrics import MetricsTracker
from .monitor import TrajectoryMonitor
from .packet import ExperiencePacketBuilder
from .results import (
    BranchContext,
    BranchMetrics,
    ExperiencePacket,
    ShareResult,
    StagnationReport,
    TrajectoryOutcome,
    TrajectoryStatus,
)
from .store import ExperienceStore

__all__ = [
    "BranchAndShareConfig",
    "BranchAndShareEngine",
    "BranchContext",
    "BranchManager",
    "BranchMetrics",
    "BranchSessionLauncher",
    "BranchStartPoint",
    "ExperiencePacket",
    "ExperiencePacketBuilder",
    "ExperienceStore",
    "FileStreamPiAdapter",
    "GitWorktreeBranchManager",
    "MemoryBranchManager",
    "MetricsTracker",
    "MockPiAdapter",
    "PiAdapter",
    "ShareResult",
    "StagnationConfig",
    "StagnationDetector",
    "StagnationReport",
    "SubprocessPiAdapter",
    "TailPiAdapter",
    "TrajectoryControl",
    "TrajectoryMonitor",
    "TrajectoryOutcome",
    "TrajectoryRunner",
    "TrajectoryStatus",
]
