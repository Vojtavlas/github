"""Launcher that creates a branch worktree and runs a Pi trajectory in it."""

import os
from typing import Optional, Tuple

from .branch_manager import BranchManager, BranchStartPoint
from .config import BranchAndShareConfig
from .control import TrajectoryControl
from .detector import StagnationDetector
from .metrics import MetricsTracker
from .monitor import TrajectoryMonitor
from .results import BranchContext, ExperiencePacket, TrajectoryOutcome
from .store import ExperienceStore


class BranchSessionLauncher:
    """Create an isolated branch and run a Pi trajectory inside it.

    This is the high-level integration point between the engine and the Pi
    agent: it wires ``BranchManager``, ``TrajectoryRunner``, and optionally
    ``ExperienceStore`` into one ``launch()`` call.
    """

    def __init__(
        self,
        config: BranchAndShareConfig,
        branch_manager: BranchManager,
        runner_factory,
        store: Optional[ExperienceStore] = None,
    ) -> None:
        self.config = config
        self.branch_manager = branch_manager
        self.runner_factory = runner_factory
        self.store = store

    def launch(
        self,
        parent: Optional[BranchContext],
        branch_id: int,
        last_packet: Optional[ExperiencePacket],
        start_point: Optional[BranchStartPoint] = None,
    ) -> Tuple[BranchContext, TrajectoryOutcome, TrajectoryControl]:
        """Create a branch, seed it with ``last_packet``, and run the agent.

        Returns the branch context, the trajectory outcome, and the control
        object that was used so the caller can build an experience packet.
        """
        if start_point is None:
            if parent is not None and self.config.reuse_checkpoints:
                start_point = BranchStartPoint.LAST_CHECKPOINT
            else:
                start_point = BranchStartPoint.ORIGINAL

        context = self.branch_manager.create_branch(
            parent, start_point, last_packet, branch_id
        )

        git_root = (
            context.worktree_path
            if os.path.isdir(context.worktree_path)
            else None
        )
        monitor = TrajectoryMonitor(git_repo_root=git_root)
        detector = StagnationDetector(self.config.stagnation)
        metrics = MetricsTracker()
        control = TrajectoryControl(monitor, detector, metrics, self.config)

        runner = self.runner_factory()
        runner.reset(context)

        metrics.start()
        outcome = runner.run(control)
        metrics.stop()

        return context, outcome, control
