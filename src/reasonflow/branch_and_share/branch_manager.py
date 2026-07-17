"""Branch isolation strategies for branch_and_share."""

import json
import os
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


class BranchManagerError(Exception):
    """Base exception for all branch manager failures."""

    def __init__(
        self,
        message: str,
        *,
        command: Optional[list[str]] = None,
        returncode: Optional[int] = None,
        stderr: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stderr = stderr

    def __str__(self) -> str:
        parts = [self.args[0]]
        if self.command is not None:
            parts.append(f"command={' '.join(self.command)!r}")
        if self.returncode is not None:
            parts.append(f"rc={self.returncode}")
        if self.stderr:
            parts.append(f"stderr={self.stderr!r}")
        return " ".join(parts)


class GitWorktreeBranchManagerError(BranchManagerError):
    """Failure from the git worktree branch manager."""


class MemoryBranchManagerError(BranchManagerError):
    """Failure from the in-memory branch manager."""


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
        self._assert_clean_repo()
        try:
            self.base_commit = self._git("rev-parse", "HEAD").strip()
        except GitWorktreeBranchManagerError as exc:
            raise GitWorktreeBranchManagerError(
                f"cannot determine base commit for {self.repo_root}",
                command=exc.command,
                returncode=exc.returncode,
                stderr=exc.stderr,
            ) from exc

    def _assert_clean_repo(self) -> None:
        """Verify the repository is a clean git checkout."""
        status = self._git(
            "status", "--porcelain", "--untracked-files=no"
        )
        if status.strip():
            raise GitWorktreeBranchManagerError(
                f"repo {self.repo_root} has uncommitted changes"
            )

    def _git(self, *args: str, cwd: Optional[str] = None) -> str:
        cwd = cwd or str(self.repo_root)
        command = ["git", *args]
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise GitWorktreeBranchManagerError(
                f"git command failed: {' '.join(command)}",
                command=command,
                returncode=exc.returncode,
                stderr=exc.stderr,
            ) from exc
        except FileNotFoundError as exc:
            raise GitWorktreeBranchManagerError(
                f"git executable not found for command: {' '.join(command)}",
                command=command,
            ) from exc
        return result.stdout

    def create_branch(
        self,
        parent: Optional[BranchContext],
        start_point: BranchStartPoint,
        packet: Optional[ExperiencePacket],
        branch_id: int,
    ) -> BranchContext:
        start_ref = self.base_commit
        if start_point == BranchStartPoint.LAST_CHECKPOINT and parent is not None:
            start_ref = parent.start_ref or start_ref

        branch_name = f"rf-attempt-{branch_id}"
        worktree_path = self.worktrees_dir / branch_name

        if self._git("branch", "--list", branch_name).strip():
            raise GitWorktreeBranchManagerError(
                f"branch {branch_name!r} already exists"
            )

        if worktree_path.exists():
            raise GitWorktreeBranchManagerError(
                f"worktree path {worktree_path} already exists"
            )

        self._git("branch", branch_name, start_ref)

        try:
            self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GitWorktreeBranchManagerError(
                f"cannot create worktrees directory {self.worktrees_dir}: {exc}"
            ) from exc

        self._git("worktree", "add", str(worktree_path), branch_name)

        if packet is not None:
            try:
                self._write_context_file(worktree_path, packet, branch_id)
            except OSError as exc:
                raise GitWorktreeBranchManagerError(
                    f"cannot write branch context file in {worktree_path}: {exc}"
                ) from exc

        start_commit = self._git(
            "rev-parse", "HEAD", cwd=str(worktree_path)
        ).strip()
        return BranchContext(
            branch_id=branch_id,
            worktree_path=str(worktree_path),
            start_ref=start_ref,
            start_commit=start_commit,
            summary="",
            base_branch=self.base_branch,
            parent_branch_id=parent.branch_id if parent else None,
            final_packet=None,
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
        try:
            (worktree_path / ".branch_context.json").write_text(
                json.dumps(context, indent=2, default=str)
            )
        except OSError as exc:
            raise GitWorktreeBranchManagerError(
                f"cannot write branch context file in {worktree_path}: {exc}"
            ) from exc

    def checkpoint(self, context: BranchContext, label: str) -> str:
        cwd = str(context.worktree_path)
        self._git("add", "-A", cwd=cwd)
        try:
            self._git("commit", "-m", f"checkpoint: {label}", cwd=cwd)
        except BranchManagerError as exc:
            if "nothing to commit" not in (exc.stderr or ""):
                raise
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
        if not isinstance(start_point, BranchStartPoint):
            raise MemoryBranchManagerError(
                f"invalid start_point {start_point!r}; "
                "must be a BranchStartPoint member"
            )
        if start_point == BranchStartPoint.LAST_CHECKPOINT:
            if parent is None:
                raise MemoryBranchManagerError(
                    "LAST_CHECKPOINT requires a parent branch"
                )
            if parent.branch_id not in self.checkpoints:
                raise MemoryBranchManagerError(
                    f"parent branch {parent.branch_id} has no checkpoint"
                )
            start_ref = self.checkpoints[parent.branch_id]
        else:
            start_ref = self.base_commit

        worktree_path = os.path.join(tempfile.gettempdir(), f"branch-{branch_id}")
        return BranchContext(
            branch_id=branch_id,
            worktree_path=worktree_path,
            start_ref=start_ref,
            start_commit=start_ref,
            summary="",
            base_branch="main",
            parent_branch_id=parent.branch_id if parent else None,
            final_packet=None,
        )

    def checkpoint(self, context: BranchContext, label: str) -> str:
        sha = f"sha-{context.branch_id}-{label}"
        self.checkpoints[context.branch_id] = sha
        return sha
