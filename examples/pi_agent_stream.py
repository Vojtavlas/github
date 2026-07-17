#!/usr/bin/env python3
"""Simulated streaming Pi agent for the branch_and_share demo.

This script is launched by ``SubprocessPiAdapter`` in a branch worktree. It
emits JSONL events to ``stdout`` with small realistic delays:

- Branch 0 repeats a failing command three times (stagnation), then emits a
  ``stagnation`` status event so the engine can branch.
- Branch 1 changes strategy, runs a passing test, and emits a ``success``
  status event.
"""

import json
import os
import time
from typing import Any, Dict


def emit(event: Dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)
    time.sleep(0.02)


def main() -> None:
    branch_id = int(os.environ.get("BRANCH_ID", "0"))

    # Tool call: read the file we may edit.
    emit({"kind": "tool_call", "name": "read_file", "args": {"path": "foo.py"}})

    if branch_id == 0:
        # Stagnation loop: same failing command repeated several times.
        for _ in range(3):
            emit(
                {
                    "kind": "command",
                    "command": "pytest -q",
                    "output": "1 failed",
                    "exit_code": 1,
                }
            )
        emit({"kind": "status", "status": "stagnation"})
    else:
        # Change strategy and succeed.
        with open("foo.py", "w") as f:
            f.write("good")
        emit(
            {
                "kind": "file_change",
                "path": "foo.py",
                "change_type": "modified",
            }
        )
        emit(
            {
                "kind": "test",
                "name": "test_foo",
                "passed": True,
                "output": "",
            }
        )
        emit({"kind": "status", "status": "success", "result": "fixed"})


if __name__ == "__main__":
    main()
