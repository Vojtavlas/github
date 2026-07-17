"""End-to-end demo of branch_and_share with git worktrees and a streaming Pi agent."""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchAndShareEngine,
    BranchSessionLauncher,
    ExperienceStore,
    GitWorktreeBranchManager,
    StagnationConfig,
    SubprocessPiAdapter,
)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path: Path) -> None:
    shutil.rmtree(str(path), onerror=_remove_readonly)


def _init_temp_repo(tmp_dir: str) -> Path:
    repo = Path(tmp_dir) / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "demo@example.com")
    _git(repo, "config", "user.name", "Demo")
    (repo / "foo.py").write_text("original")
    (repo / "README.md").write_text("demo repo")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def _agent_script(tmp_dir: str) -> Path:
    script = Path(tmp_dir) / "pi_agent_stream.py"
    source = Path(__file__).with_name("pi_agent_stream.py")
    if source.exists():
        shutil.copy2(str(source), str(script))
    else:
        # Fallback for environments where the example file is not adjacent.
        script.write_text(
            '''import json, os, time
def emit(event):
    print(json.dumps(event), flush=True)
    time.sleep(0.02)

branch_id = int(os.environ.get("BRANCH_ID", "0"))
emit({"kind": "tool_call", "name": "read_file", "args": {"path": "foo.py"}})

if branch_id == 0:
    with open("foo.py", "w") as f:
        f.write("bad")
    for _ in range(3):
        emit({"kind": "command", "command": "pytest -q", "output": "1 failed", "exit_code": 1})
    emit({"kind": "status", "status": "stagnation"})
else:
    with open("foo.py", "w") as f:
        f.write("good")
    emit({"kind": "file_change", "path": "foo.py", "change_type": "modified"})
    emit({"kind": "test", "name": "test_foo", "passed": True, "output": ""})
    emit({"kind": "status", "status": "success", "result": "fixed"})
'''
        )
    return script


def _cleanup_worktrees(repo: Path) -> None:
    for i in range(10):
        wt = repo / ".worktrees" / f"rf-attempt-{i}"
        if wt.exists():
            try:
                _git(repo, "worktree", "remove", "--force", str(wt), check=False)
            except Exception:
                pass
        try:
            _git(repo, "branch", "-D", f"rf-attempt-{i}", check=False)
        except Exception:
            pass


def _runner_factory(script: Path):
    return SubprocessPiAdapter([sys.executable, str(script)])


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="branch_and_share_demo_")
    try:
        repo = _init_temp_repo(tmp_dir)
        script = _agent_script(tmp_dir)
        store_path = Path(tmp_dir) / "experience.jsonl"

        config = BranchAndShareConfig(
            max_branches=2,
            stagnation=StagnationConfig(repeat_threshold=2),
            use_git_worktrees=True,
            reuse_checkpoints=True,
        )
        branch_manager = GitWorktreeBranchManager(
            repo_root=str(repo), base_branch="master", worktrees_dir=".worktrees"
        )
        store = ExperienceStore(store_path)
        launcher = BranchSessionLauncher(
            config,
            branch_manager,
            lambda: _runner_factory(script),
            store=store,
        )
        engine = BranchAndShareEngine(
            config,
            branch_manager,
            lambda: _runner_factory(script),
            launcher=launcher,
            store=store,
        )

        result = engine.solve()

        print("=== ShareResult ===")
        print(f"success={result.success}")
        print(f"best_branch_id={result.best_branch_id}")
        print(f"branches={len(result.branches)}")
        print(f"metrics={result.metrics}")
        print()

        if result.final_packet:
            print("=== Final Experience Packet ===")
            print(
                f"current_passing_tests={result.final_packet.current_passing_tests}"
            )
            print(
                f"current_failing_tests={result.final_packet.current_failing_tests}"
            )
            print(
                "recommended_next_actions="
                f"{result.final_packet.recommended_next_actions}"
            )
            diff = result.final_packet.modified_files_and_diff
            print(f"modified_files_and_diff (first 500 chars)={diff[:500]}")
            print()

        latest_worktree = Path(result.branches[-1].worktree_path)
        context_file = latest_worktree / ".branch_context.json"
        if context_file.exists():
            print("=== .branch_context.json in latest worktree ===")
            print(context_file.read_text())
        else:
            print("No .branch_context.json found in latest worktree.")

        session_files = list((repo / ".reasonflow" / "sessions").glob("*.jsonl"))
        if session_files:
            print("=== Session log ===")
            print(f"path={session_files[0]}")
            lines = session_files[0].read_text(encoding="utf-8").strip().splitlines()
            print(f"lines={len(lines)}")
            if lines:
                print(f"first_event={lines[0][:200]}")
        else:
            print("No session log found.")

        _cleanup_worktrees(repo)
    finally:
        _rmtree(Path(tmp_dir))


if __name__ == "__main__":
    main()
