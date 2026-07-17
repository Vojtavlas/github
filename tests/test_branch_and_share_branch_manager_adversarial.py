"""Adversarial tests for branch_and_share branch managers."""

import shutil
import subprocess
from pathlib import Path

import pytest

from reasonflow.branch_and_share import (
    BranchManagerError,
    BranchStartPoint,
    GitWorktreeBranchManager,
    MemoryBranchManager,
)

git_installed = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed"
)


def _init_git_repo(path: Path, with_commit: bool = True) -> None:
    """Create a minimal git repo, optionally with an initial commit."""
    subprocess.run(["git", "init"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    if with_commit:
        (path / "README.md").write_text("hello")
        subprocess.run(
            ["git", "add", "."], cwd=str(path), check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(path),
            check=True,
            capture_output=True,
        )


def _raise_file_not_found(*_args, **_kwargs):
    raise FileNotFoundError(2, "No such file or directory: 'git'")


@git_installed
def test_git_worktree_not_a_git_directory(tmp_path: Path) -> None:
    with pytest.raises(BranchManagerError) as exc_info:
        GitWorktreeBranchManager(repo_root=str(tmp_path))
    assert "not a git repository" in str(exc_info.value)
    assert exc_info.value.returncode == 128
    assert exc_info.value.stderr is not None
    assert "not a git repository" in (exc_info.value.stderr or "").lower()


@git_installed
def test_git_worktree_dirty_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("dirty")

    with pytest.raises(BranchManagerError, match="uncommitted changes"):
        GitWorktreeBranchManager(repo_root=str(tmp_path))


@git_installed
def test_git_worktree_missing_base_commit_ref(tmp_path: Path) -> None:
    _init_git_repo(tmp_path, with_commit=False)

    with pytest.raises(BranchManagerError) as exc_info:
        GitWorktreeBranchManager(repo_root=str(tmp_path))
    assert "cannot determine base commit" in str(exc_info.value)
    assert exc_info.value.returncode == 128


@git_installed
def test_git_worktree_missing_start_ref_at_create_branch(tmp_path: Path) -> None:
    """A base_commit that exists at init but is invalid when branching."""
    _init_git_repo(tmp_path)
    manager = GitWorktreeBranchManager(repo_root=str(tmp_path))
    manager.base_commit = "0" * 40

    with pytest.raises(BranchManagerError) as exc_info:
        manager.create_branch(
            None, BranchStartPoint.ORIGINAL, None, branch_id=0
        )
    assert exc_info.value.returncode == 128
    assert "not a valid" in (exc_info.value.stderr or "").lower()


@git_installed
def test_git_worktree_existing_worktree_directory(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    manager = GitWorktreeBranchManager(repo_root=str(tmp_path))
    manager.create_branch(None, BranchStartPoint.ORIGINAL, None, branch_id=1)

    (manager.worktrees_dir / "rf-attempt-2").mkdir(parents=True, exist_ok=True)

    with pytest.raises(BranchManagerError, match="worktree path .* already exists"):
        manager.create_branch(None, BranchStartPoint.ORIGINAL, None, branch_id=2)


@git_installed
def test_git_worktree_existing_branch(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    subprocess.run(
        ["git", "branch", "rf-attempt-0"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    manager = GitWorktreeBranchManager(repo_root=str(tmp_path))

    with pytest.raises(BranchManagerError, match="branch 'rf-attempt-0' already exists"):
        manager.create_branch(None, BranchStartPoint.ORIGINAL, None, branch_id=0)


@git_installed
def test_git_worktree_git_not_in_path(tmp_path: Path, monkeypatch) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setattr(
        "reasonflow.branch_and_share.branch_manager.subprocess.run",
        _raise_file_not_found,
    )

    with pytest.raises(BranchManagerError) as exc_info:
        GitWorktreeBranchManager(repo_root=str(tmp_path))
    assert "git executable not found" in str(exc_info.value)
    assert exc_info.value.command is not None
    assert exc_info.value.command[0] == "git"


@git_installed
def test_git_worktree_read_only_repo(tmp_path: Path) -> None:
    """Simulate a write failure by putting a file where a directory is needed."""
    _init_git_repo(tmp_path)
    (tmp_path / "blocker").write_text("not a directory")
    manager = GitWorktreeBranchManager(
        repo_root=str(tmp_path), worktrees_dir="blocker/wt"
    )

    with pytest.raises(BranchManagerError, match="cannot create worktrees directory"):
        manager.create_branch(None, BranchStartPoint.ORIGINAL, None, branch_id=0)


def test_memory_branch_manager_invalid_start_point() -> None:
    manager = MemoryBranchManager()
    with pytest.raises(BranchManagerError, match="invalid start_point"):
        manager.create_branch(None, "not_a_member", None, branch_id=0)  # type: ignore[arg-type]


def test_memory_branch_manager_missing_parent() -> None:
    manager = MemoryBranchManager()
    with pytest.raises(BranchManagerError, match="LAST_CHECKPOINT requires a parent"):
        manager.create_branch(
            None, BranchStartPoint.LAST_CHECKPOINT, None, branch_id=0
        )


def test_memory_branch_manager_missing_checkpoint() -> None:
    manager = MemoryBranchManager()
    parent = manager.create_branch(
        None, BranchStartPoint.ORIGINAL, None, branch_id=0
    )
    with pytest.raises(BranchManagerError, match="parent branch 0 has no checkpoint"):
        manager.create_branch(
            parent, BranchStartPoint.LAST_CHECKPOINT, None, branch_id=1
        )
