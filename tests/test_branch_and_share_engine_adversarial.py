"""Adversarial / chaos tests for BranchAndShareEngine and BranchSessionLauncher."""

from pathlib import Path
from typing import Optional

import pytest

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchAndShareEngine,
    BranchContext,
    BranchStartPoint,
    ExperiencePacket,
    ExperienceStore,
    MemoryBranchManager,
    MockPiAdapter,
    ShareResult,
    StagnationConfig,
    TrajectoryControl,
    TrajectoryOutcome,
    TrajectoryRunner,
    TrajectoryStatus,
)


def _small_config(max_branches: int = 2) -> BranchAndShareConfig:
    return BranchAndShareConfig(
        max_branches=max_branches,
        stagnation=StagnationConfig(
            repeat_threshold=3,
            file_read_threshold=3,
            test_window=2,
            tool_failure_threshold=2,
            churn_threshold=3,
            token_limit=100000,
            token_warn_fraction=0.85,
            window=10,
        ),
    )


class SuccessRunner(TrajectoryRunner):
    """A runner that always reports success."""

    def reset(self, context: BranchContext) -> None:
        pass

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        return TrajectoryOutcome(status=TrajectoryStatus.SUCCESS, result={"ok": True})


class RunReturnsNoneRunner(TrajectoryRunner):
    """A runner whose run() returns None."""

    def reset(self, context: BranchContext) -> None:
        pass

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:  # type: ignore[return]
        return None  # type: ignore[return]


class RunRaisesRunner(TrajectoryRunner):
    """A runner whose run() raises."""

    def reset(self, context: BranchContext) -> None:
        pass

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        raise ValueError("run exploded")


class ResetRaisesRunner(TrajectoryRunner):
    """A runner whose reset() raises."""

    def reset(self, context: BranchContext) -> None:
        raise ValueError("reset exploded")

    def run(self, control: TrajectoryControl) -> TrajectoryOutcome:
        return TrajectoryOutcome(status=TrajectoryStatus.SUCCESS, result={"ok": True})


def _failing_factory() -> TrajectoryRunner:
    raise RuntimeError("factory exploded")


class CreateBranchFailManager(MemoryBranchManager):
    """A branch manager whose create_branch always raises."""

    def create_branch(
        self,
        parent: Optional[BranchContext],
        start_point: BranchStartPoint,
        packet: Optional[ExperiencePacket],
        branch_id: int,
    ) -> BranchContext:
        raise OSError("cannot create worktree")


class ReadOnlyStore(ExperienceStore):
    """Simulate a read-only store by raising on append."""

    def append(self, packet: ExperiencePacket) -> None:
        raise OSError("read-only file")


def test_runner_factory_raises_is_not_fatal():
    config = _small_config(max_branches=2)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, _failing_factory)

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 2
    assert result.metrics.branch_count == 2
    assert result.final_packet is not None


def test_runner_run_returns_none_is_not_fatal():
    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(
        config, manager, lambda: RunReturnsNoneRunner()
    )

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 1
    assert result.metrics.branch_count == 1
    assert result.final_packet is not None


def test_runner_run_raises_is_not_fatal():
    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, lambda: RunRaisesRunner())

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 1
    assert result.metrics.branch_count == 1
    assert result.final_packet is not None


def test_runner_reset_raises_is_not_fatal():
    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(
        config, manager, lambda: ResetRaisesRunner()
    )

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 1
    assert result.metrics.branch_count == 1
    assert result.final_packet is not None


def test_create_branch_raises_is_not_fatal():
    config = _small_config(max_branches=2)
    manager = CreateBranchFailManager()
    engine = BranchAndShareEngine(config, manager, lambda: SuccessRunner())

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 2
    assert result.metrics.branch_count == 2
    assert result.final_packet is not None


def test_store_append_raises_is_not_fatal(tmp_path: Path):
    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    store = ReadOnlyStore(str(tmp_path / "store.jsonl"))
    engine = BranchAndShareEngine(
        config,
        manager,
        lambda: MockPiAdapter(scenario=[{"kind": "test", "name": "t1", "passed": True}]),
        store=store,
    )

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert result.success
    assert result.best_branch_id == 0
    assert len(result.branches) == 1
    assert result.final_packet is not None


def test_max_branches_zero_returns_empty_result():
    config = _small_config(max_branches=0)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(config, manager, lambda: SuccessRunner())

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert result.branches == []
    assert result.metrics.branch_count == 0
    assert result.final_packet is None


def test_max_branches_one_stagnation():
    config = _small_config(max_branches=1)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(
        config,
        manager,
        lambda: MockPiAdapter(
            scenario=[{"kind": "file_read", "path": "stuck.py"}] * 3
        ),
    )

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 1
    assert result.metrics.branch_count == 1
    assert result.final_packet is not None


def test_config_allows_zero_and_one_branches():
    assert BranchAndShareConfig(max_branches=0).max_branches == 0
    assert BranchAndShareConfig(max_branches=1).max_branches == 1
    with pytest.raises(ValueError):
        BranchAndShareConfig(max_branches=-1)


class _RaisingLauncher:
    """A launcher whose launch() always raises."""

    def launch(self, *args, **kwargs):
        raise RuntimeError("launcher boom")


def test_launcher_launch_raises_is_not_fatal():
    config = _small_config(max_branches=2)
    manager = MemoryBranchManager()
    engine = BranchAndShareEngine(
        config,
        manager,
        lambda: SuccessRunner(),
        launcher=_RaisingLauncher(),
    )

    result = engine.solve()

    assert isinstance(result, ShareResult)
    assert not result.success
    assert result.best_branch_id is None
    assert len(result.branches) == 2
    assert result.metrics.branch_count == 2
    assert result.final_packet is not None
