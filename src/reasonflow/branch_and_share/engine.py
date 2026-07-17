"""Branch-and-share engine for failure-aware Pi coding-agent trajectories."""

import time
from typing import Callable, List, Optional

from .adapter import TrajectoryRunner
from .branch_manager import BranchManager, BranchStartPoint
from .config import BranchAndShareConfig
from .launcher import BranchSessionLauncher
from .packet import ExperiencePacketBuilder
from .results import (
    BranchContext,
    BranchMetrics,
    ExperiencePacket,
    ShareResult,
    TrajectoryStatus,
)
from .store import ExperienceStore


class BranchAndShareEngine:
    """Run a Pi coding-agent trajectory, detect stagnation, branch, and share."""

    def __init__(
        self,
        config: BranchAndShareConfig,
        branch_manager: BranchManager,
        runner_factory: Callable[[], TrajectoryRunner],
        launcher: Optional[BranchSessionLauncher] = None,
        store: Optional[ExperienceStore] = None,
    ) -> None:
        self.config = config
        self.branch_manager = branch_manager
        self.runner_factory = runner_factory
        self.launcher = launcher or BranchSessionLauncher(
            config, branch_manager, runner_factory, store
        )
        self.store = store
        self.branches: List[BranchContext] = []
        self.cumulative = BranchMetrics()

    def solve(self) -> ShareResult:
        """Run up to ``max_branches`` trajectories, branching on stagnation."""
        start = time.time()
        last_packet: Optional[ExperiencePacket] = None

        # Seed the first branch from any persisted experience.
        if self.store is not None:
            recent = self.store.load_recent(1)
            if recent:
                last_packet = recent[-1]

        for branch_id in range(self.config.max_branches):
            parent = self.branches[-1] if self.branches else None
            start_point = (
                BranchStartPoint.LAST_CHECKPOINT
                if (parent and self.config.reuse_checkpoints)
                else BranchStartPoint.ORIGINAL
            )

            context, outcome, control = self.launcher.launch(
                parent, branch_id, last_packet, start_point
            )
            self.branches.append(context)

            branch_metrics = control.metrics.snapshot(
                branch_count=branch_id + 1,
                final_success=(outcome.status == TrajectoryStatus.SUCCESS),
            )
            self._accumulate(branch_metrics)

            packet = ExperiencePacketBuilder(
                control.monitor, context, branch_metrics
            ).build(outcome.report)
            context.final_packet = packet
            last_packet = packet

            if self.store is not None:
                self.store.append(packet)

            if outcome.status == TrajectoryStatus.SUCCESS:
                self.cumulative.final_success = True
                self.cumulative.wall_clock_ms = (time.time() - start) * 1000.0
                return ShareResult(
                    best_branch_id=context.branch_id,
                    branches=self.branches,
                    success=True,
                    metrics=self.cumulative,
                    final_packet=packet,
                )

            checkpoint_ref = self.branch_manager.checkpoint(
                context, f"stuck-{branch_id}"
            )
            context.start_ref = checkpoint_ref

        self.cumulative.wall_clock_ms = (time.time() - start) * 1000.0
        return ShareResult(
            best_branch_id=None,
            branches=self.branches,
            success=False,
            metrics=self.cumulative,
            final_packet=last_packet,
        )

    def _accumulate(self, m: BranchMetrics) -> None:
        self.cumulative.total_tokens += m.total_tokens
        self.cumulative.model_calls += m.model_calls
        self.cumulative.tool_calls += m.tool_calls
        self.cumulative.duplicated_work += m.duplicated_work
        self.cumulative.branch_count = m.branch_count
        self.cumulative.tests_passing = m.tests_passing
        self.cumulative.tests_failing = m.tests_failing
