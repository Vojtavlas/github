"""Branch isolation strategies for branch_and_share."""

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from .results import BranchContext, ExperiencePacket


class BranchStartPoint(str, Enum):
    ORIGINAL = "original"
    LAST_CHECKPOINT = "last_checkpoint"


class BranchManager(ABC):
    """Abstract strategy for creating and checkpointing branches."""

    @abstractmethod
    def create_branch(
        self,
        parent: Optional[BranchContext],
        start_point: BranchStartPoint,
        packet: Optional[ExperiencePacket],
        branch_id: int,
    ) -> BranchContext:
        """Create a new branch and return its context."""

    @abstractmethod
    def checkpoint(self, context: BranchContext, label: str) -> str:
        """Persist the current branch state and return a reference."""


class GitWorktreeBranchManager(BranchManager):
    """Isolate attempts using git worktrees."""

    def __init__(
        self,
        repo_root: str,
        base_branch: str = "main",
        worktrees_dir: str = ".worktrees",
    ) -> None:
        self.repo_root = Path(repo_root)
        self.base_branch = base_branch
        self.worktrees_dir = self.repo_root / worktrees_dir
        self.base_commit = self._git("rev-parse", "HEAD").strip()

    def _git(self, *args: str, cwd: Optional[str] = None) -> str:
        cwd = cwd or str(self.repo_root)
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def create_branch(
        self,
        parent: Optional[BranchContext],
        start_point: BranchStartPoint,
        packet: Optional[ExperiencePacket],
        branch_id: int,
    ) -> BranchContext:
        start_ref = self.base_commit
        if start_point == BranchStartPoint.LAST_CHECKPOINT and parent:
            start_ref = parent.start_ref or start_ref

        branch_name = f"rf-attempt-{branch_id}"
        worktree_path = self.worktrees_dir / branch_name

        self._git("branch", "-f", branch_name, start_ref)

        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        if worktree_path.exists():
            shutil.rmtree(worktree_path)
        self._git("worktree", "add", str(worktree_path), branch_name)

        if packet is not None:
            self._write_context_file(worktree_path, packet, branch_id)

        start_commit = self._git("rev-parse", "HEAD", cwd=str(worktree_path)).strip()
        return BranchContext(
            branch_id=branch_id,
            worktree_path=str(worktree_path),
            start_ref=start_ref,
            start_commit=start_commit,
            summary="",
            base_branch=self.base_branch,
            parent_branch_id=parent.branch_id if parent else None,
        )

    def _write_context_file(
        self, worktree_path: Path, packet: ExperiencePacket, branch_id: int
    ) -> None:
        context = {
            "branch_id": branch_id,
            "recommended_next_actions": packet.recommended_next_actions,
            "current_failing_tests": packet.current_failing_tests,
            "evidence_of_failure": packet.evidence_of_failure[:5],
            "hypotheses": [h.description for h in packet.hypotheses_attempted],
            "worktree_path": str(worktree_path),
        }
        (worktree_path / ".branch_context.json").write_text(
            json.dumps(context, indent=2, default=str)
        )

    def checkpoint(self, context: BranchContext, label: str) -> str:
        cwd = str(context.worktree_path)
        self._git("add", "-A", cwd=cwd)
        try:
            self._git("commit", "-m", f"checkpoint: {label}", cwd=cwd)
        except subprocess.CalledProcessError:
            pass
        return self._git("rev-parse", "HEAD", cwd=cwd).strip()


class MemoryBranchManager(BranchManager):
    """In-memory branch manager for fast tests."""

    def __init__(self, base_commit: str = "abc123") -> None:
        self.base_commit = base_commit
        self.checkpoints: Dict[int, str] = {}

    def create_branch(
        self,
        parent: Optional[BranchContext],
        start_point: BranchStartPoint,
        packet: Optional[ExperiencePacket],
        branch_id: int,
    ) -> BranchContext:
        start_ref = self.base_commit
        if start_point == BranchStartPoint.LAST_CHECKPOINT and parent:
            start_ref = self.checkpoints.get(parent.branch_id, start_ref)

        worktree_path = os.path.join(tempfile.gettempdir(), f"branch-{branch_id}")
        return BranchContext(
            branch_id=branch_id,
            worktree_path=worktree_path,
            start_ref=start_ref,
            start_commit=start_ref,
            summary="",
            base_branch="main",
            parent_branch_id=parent.branch_id if parent else None,
        )

    def checkpoint(self, context: BranchContext, label: str) -> str:
        sha = f"sha-{context.branch_id}-{label}"
        self.checkpoints[context.branch_id] = sha
        return sha
