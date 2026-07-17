"""Branch-and-share engine for failure-aware Pi coding-agent trajectories."""

import time
from typing import Callable, List, Optional, Tuple

from .adapter import TrajectoryRunner
from .branch_manager import BranchManager, BranchStartPoint
from .config import BranchAndShareConfig
from .control import TrajectoryControl
from .detector import StagnationDetector
from .launcher import BranchSessionLauncher
from .logging import BranchSessionLogger
from .metrics import MetricsTracker
from .monitor import TrajectoryMonitor
from .packet import ExperiencePacketBuilder
from .results import (
    BranchContext,
    BranchMetrics,
    BranchSessionReport,
    ExperiencePacket,
    ShareResult,
    TrajectoryOutcome,
    TrajectoryStatus,
)
from .store import ExperienceStore

_FAILED_WORKTREE = "<launch-failed>"


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
        self._session_logger: Optional[BranchSessionLogger] = None
        self._last_outcome: Optional[TrajectoryOutcome] = None

    def _create_logger(self) -> Optional[BranchSessionLogger]:
        """Create a session logger rooted in the repo or configured path."""
        repo_root = self.config.log_repo_root
        if repo_root is None:
            repo_root = getattr(self.branch_manager, "repo_root", None)
        if repo_root is None:
            return None
        return BranchSessionLogger(repo_root=str(repo_root))

    def _safe_launch(
        self,
        parent: Optional[BranchContext],
        branch_id: int,
        last_packet: Optional[ExperiencePacket],
        start_point: BranchStartPoint,
    ) -> Tuple[BranchContext, TrajectoryOutcome, TrajectoryControl]:
        """Invoke the launcher and return a synthetic error branch if it raises."""
        try:
            return self.launcher.launch(
                parent,
                branch_id,
                last_packet,
                start_point,
                logger=self._session_logger,
            )
        except Exception as exc:
            context = BranchContext(
                branch_id=branch_id,
                worktree_path=_FAILED_WORKTREE,
                start_ref="",
                start_commit="",
                summary=f"launcher.launch() failed: {exc}",
                base_branch=self.config.base_branch,
                parent_branch_id=parent.branch_id if parent else None,
            )
            monitor = TrajectoryMonitor(git_repo_root=_FAILED_WORKTREE)
            detector = StagnationDetector(self.config.stagnation)
            metrics = MetricsTracker()
            control = TrajectoryControl(monitor, detector, metrics, self.config)
            metrics.start()
            metrics.stop()
            return (
                context,
                TrajectoryOutcome(
                    status=TrajectoryStatus.ERROR,
                    error=f"launcher.launch() raised: {exc}",
                ),
                control,
            )

    def solve(self) -> ShareResult:
        """Run up to ``max_branches`` trajectories, branching on stagnation."""
        start = time.time()
        self._session_logger = self._create_logger()
        last_packet: Optional[ExperiencePacket] = None

        def _close_logger() -> None:
            if self._session_logger is not None:
                self._session_logger.close()
                self._session_logger = None

        # Seed the first branch from any persisted experience.
        if self.store is not None:
            try:
                recent = self.store.load_recent(1)
                if recent:
                    last_packet = recent[-1]
            except Exception:
                # Persisted experience is optional; a failing store should not
                # prevent the engine from attempting a fresh trajectory.
                pass

        try:
            for branch_id in range(self.config.max_branches):
                try:
                    parent = self.branches[-1] if self.branches else None
                    start_point = (
                        BranchStartPoint.LAST_CHECKPOINT
                        if (parent and self.config.reuse_checkpoints)
                        else BranchStartPoint.ORIGINAL
                    )

                    context, outcome, control = self._safe_launch(
                        parent, branch_id, last_packet, start_point
                    )
                    self._last_outcome = outcome
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
                        try:
                            self.store.append(packet)
                        except Exception:
                            # Storing experience is best-effort; the trajectory
                            # outcome still determines overall success.
                            pass

                    if outcome.status == TrajectoryStatus.SUCCESS:
                        self.cumulative.final_success = True
                        self.cumulative.wall_clock_ms = (
                            time.time() - start
                        ) * 1000.0
                        _close_logger()
                        return ShareResult(
                            best_branch_id=context.branch_id,
                            branches=self.branches,
                            success=True,
                            metrics=self.cumulative,
                            final_packet=packet,
                        )

                    try:
                        checkpoint_ref = self.branch_manager.checkpoint(
                            context, f"stuck-{branch_id}"
                        )
                        context.start_ref = checkpoint_ref
                    except Exception:
                        # If checkpointing fails, continue with the existing
                        # start_ref.
                        pass
                except Exception:
                    # Non-fatal branch failure: continue to the next branch
                    # rather than aborting the whole solve.
                    continue

            self.cumulative.wall_clock_ms = (time.time() - start) * 1000.0
            _close_logger()
            return ShareResult(
                best_branch_id=None,
                branches=self.branches,
                success=False,
                metrics=self.cumulative,
                final_packet=last_packet,
            )
        except Exception:
            self.cumulative.wall_clock_ms = (time.time() - start) * 1000.0
            _close_logger()
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

    def report(self) -> BranchSessionReport:
        """Return a summary report of the completed session."""
        final_outcome: Optional[TrajectoryStatus] = None
        if self._last_outcome is not None:
            final_outcome = self._last_outcome.status
        elif self.cumulative.final_success:
            final_outcome = TrajectoryStatus.SUCCESS
        return BranchSessionReport(
            branches=list(self.branches),
            total_time_ms=self.cumulative.wall_clock_ms,
            pass_count=self.cumulative.tests_passing,
            fail_count=self.cumulative.tests_failing,
            final_success=self.cumulative.final_success,
            final_outcome=final_outcome,
            metrics=self.cumulative,
        )
