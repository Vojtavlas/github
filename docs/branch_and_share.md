# `branch_and_share` protocol and adapter guide

`reasonflow.branch_and_share` runs a Pi coding agent as a sequence of isolated
branches. Each branch emits JSON events that describe what the agent is doing;
the engine detects stagnation, creates a fresh branch, and seeds it with a
portable summary of the previous branch's experience.

This document is for Pi-agent authors who want to emit compatible events or
implement a custom `PiAdapter`.

---

## 1. JSON event protocol

Each event is one line of JSON (newline-delimited JSON, NDJSON) with a `kind`
field. The validator lives in `src/reasonflow/branch_and_share/protocol.py`.

| `kind` | Required fields | Optional fields | Meaning |
|--------|-----------------|-----------------|---------|
| `tool_call` | `name` | `args`, `result`, `failed`, `tokens` | A tool was invoked |
| `file_read` | `path` | `symbols` | A file was read |
| `command` | `command` | `output`, `exit_code` | A shell command ran |
| `test` | `name` | `passed`, `output` | A test ran |
| `file_change` | `path` | `change_type`, `old_hash`, `new_hash` | A file changed |
| `model_call` | `tokens` | — | A model call consumed tokens |
| `token_usage` | `n` | — | Raw token-usage delta |
| `hypothesis` | `description`, `action`, `expected` | `observed`, `failed` | A hypothesis was tried |
| `discovery` | `text` | — | Something useful was learned |
| `status` | `status` | `result`, `error` | Terminal outcome of the branch |

`status` must be one of `success`, `error`, or `stagnation`.

### Examples

```json
{"kind":"tool_call","name":"read","args":{"path":"src/app.py"},"result":"...","failed":false,"tokens":120}
```

```json
{"kind":"file_read","path":"src/app.py","symbols":["main"]}
```

```json
{"kind":"command","command":"pytest","output":"2 passed, 1 failed","exit_code":1}
```

```json
{"kind":"test","name":"test_login","passed":false,"output":"AssertionError: 401"}
```

```json
{"kind":"file_change","path":"src/app.py","change_type":"modified","old_hash":"abc123","new_hash":"def456"}
```

```json
{"kind":"model_call","tokens":1500}
```

```json
{"kind":"token_usage","n":1500}
```

```json
{"kind":"hypothesis","description":"missing auth header","action":"add Authorization header","expected":"test_login passes","observed":"still 401","failed":true}
```

```json
{"kind":"discovery","text":"login route expects 'Bearer ' prefix"}
```

```json
{"kind":"status","status":"success","result":{"message":"done"}}
```

```json
{"kind":"status","status":"error","error":"pytest not found"}
```

```json
{"kind":"status","status":"stagnation"}
```

---

## 2. Implementing a custom `PiAdapter`

The seam is `TrajectoryRunner` in `src/reasonflow/branch_and_share/adapter.py`.

```python
from reasonflow.branch_and_share.adapter import PiAdapter
from reasonflow.branch_and_share.results import TrajectoryOutcome, TrajectoryStatus

class MyPiAdapter(PiAdapter):
    def reset(self, context):
        # context is a BranchContext with worktree_path, branch_id, etc.
        super().reset(context)
        # Load any context file written by the branch manager.
        context_file = Path(context.worktree_path) / ".branch_context.json"
        if context_file.exists():
            self.context_data = json.loads(context_file.read_text())
        else:
            self.context_data = {}

    def run(self, control) -> TrajectoryOutcome:
        # Agent loop: do work, call control.record_* methods, then
        # periodically check control.check_stagnation().
        for step in self.plan():
            control.record_tool_call(
                name=step["tool"],
                args=step.get("args", {}),
                result=step.get("result"),
                failed=step.get("failed", False),
                tokens=step.get("tokens", 0),
            )
            report = control.check_stagnation()
            if report is not None:
                return TrajectoryOutcome(
                    status=TrajectoryStatus.STAGNATION,
                    report=report,
                )

        return TrajectoryOutcome(
            status=TrajectoryStatus.SUCCESS,
            result={"message": "completed"},
        )
```

`TrajectoryControl` (`src/reasonflow/branch_and_share/control.py`) provides:

- `record_tool_call(name, args=None, result=None, failed=False, tokens=0)`
- `record_file_read(path, symbols=None)`
- `record_command(command, output="", exit_code=0)`
- `record_test_result(name, passed, output="")`
- `record_model_call(tokens=0)`
- `record_token_usage(n)`
- `record_file_change(path, change_type="modified", old_hash=None, new_hash=None)`
- `record_hypothesis(description, action, expected_test_change, observed="", failed=True)`
- `record_discovery(text)`
- `check_stagnation() -> Optional[StagnationReport]`
- `current_git_state()`

A runner returns `TrajectoryOutcome(status=TrajectoryStatus.SUCCESS|ERROR|STAGNATION, ...)`.

---

## 3. How the engine branches and seeds the next attempt

`BranchAndShareEngine.solve` (`src/reasonflow/branch_and_share/engine.py`) runs
up to `max_branches` trajectories in a loop.

### Seeding the first branch from persisted experience

Before the loop the engine calls `store.load_recent(1)` if an `ExperienceStore`
was provided. The most recent `ExperiencePacket` becomes the seed for the first
branch.

### Branching on stagnation

The runner decides when to stop by calling `control.check_stagnation()`.
`StagnationDetector` (`src/reasonflow/branch_and_share/detector.py`) analyzes
the monitor history and returns a `StagnationReport` when enough signals fire:

| Signal | What triggers it |
|--------|------------------|
| `repeated_command` | The same command appears `>= repeat_threshold` times in the last `window` commands |
| `repeated_file_read` | The same file is read `>= file_read_threshold` times in the last `window` reads |
| `repeated_tool_failure` | The same tool fails `>= tool_failure_threshold` times in the last `window` tool calls |
| `no_test_improvement` | No new tests pass over the last `test_window` test results while failures remain |
| `code_churn` | A file is changed `>= churn_threshold` times or flips back to a previous hash |
| `approaching_token_limit` | Total tokens reach `token_limit * token_warn_fraction` |

Confidence is `0.3 + 0.15 * len(signals)`, capped at `1.0`.

### Seeding the next branch from the current one

After each branch finishes the engine builds an `ExperiencePacket` via
`ExperiencePacketBuilder` (`src/reasonflow/branch_and_share/packet.py`). The
packet contains:

- `files_and_symbols_inspected`
- `commands_and_tests_run`
- `modified_files_and_diff`
- `current_passing_tests` / `current_failing_tests`
- `hypotheses_attempted`
- `evidence_of_failure`
- `useful_discoveries`
- `recommended_next_actions`
- `metrics`

That packet is passed to `BranchSessionLauncher.launch` for the next branch.
`BranchManager.create_branch` writes it to `<worktree>/.branch_context.json`
so the new agent can read recommendations and known failures at startup. The
packet is also appended to `ExperienceStore` (`src/reasonflow/branch_and_share/store.py`)
for persistence.

### Start point selection

The engine picks the start point for each new branch:

- `BranchStartPoint.LAST_CHECKPOINT` if there is a parent branch and
  `config.reuse_checkpoints` is `True`
- `BranchStartPoint.ORIGINAL` otherwise

The branch manager uses this to decide which git ref to start from.

---

## 4. Minimal example: subprocess agent

A subprocess agent only needs to print NDJSON events to `stdout` and emit a
terminal `status` event. The `SubprocessPiAdapter` launches it in the branch
worktree and replays the events into `TrajectoryControl`.

```python
#!/usr/bin/env python3
import json, os, sys

context_path = os.environ.get("BRANCH_CONTEXT_PATH")
if context_path and os.path.exists(context_path):
    ctx = json.loads(open(context_path).read())
    print(json.dumps({"kind":"discovery","text":f"prior actions: {ctx.get('recommended_next_actions',[])}"}))

# Simulate reading a file and running a test.
print(json.dumps({"kind":"file_read","path":"src/app.py","symbols":["login"]}))
print(json.dumps({"kind":"command","command":"pytest","output":"1 failed","exit_code":1}))
print(json.dumps({"kind":"test","name":"test_login","passed":False,"output":"401"}))
print(json.dumps({"kind":"discovery","text":"missing Bearer prefix"}))
print(json.dumps({"kind":"status","status":"stagnation"}))
```

Run it through the engine:

```python
from reasonflow.branch_and_share.adapter import SubprocessPiAdapter
from reasonflow.branch_and_share.branch_manager import MemoryBranchManager
from reasonflow.branch_and_share.config import BranchAndShareConfig
from reasonflow.branch_and_share.engine import BranchAndShareEngine

config = BranchAndShareConfig(max_branches=3)
manager = MemoryBranchManager()
engine = BranchAndShareEngine(
    config,
    manager,
    runner_factory=lambda: SubprocessPiAdapter(["python3", "agent.py"]),
)
result = engine.solve()
print(result)
```

---

## 5. Session logging and reporting

`BranchAndShareEngine` can write a JSON-lines session log to
`<repo_root>/.reasonflow/sessions/<iso-timestamp>.jsonl`. Enable it by setting
`log_repo_root` on `BranchAndShareConfig` or by using a `BranchManager` that
exposes a `repo_root` attribute (such as `GitWorktreeBranchManager`). When no
repo root is available, logging is silently disabled so fast and in-memory
tests remain unaffected.

Each log line contains:

| Field | Meaning |
|-------|---------|
| `branch_id` | Branch that produced the event |
| `kind` | Event kind (`branch_start`, `tool_call`, `file_read`, `command`, `test`, `file_change`, `model_call`, `token_usage`, `hypothesis`, `discovery`, `status`) |
| `timestamp` | Unix timestamp when the line was written |
| `elapsed_ms` | Milliseconds since the branch started |
| `stagnation_signals` | Active stagnation signals at that moment |
| `outcome` | Terminal status string, only on `status` lines |
| `payload` | Event-specific details (tool name, file path, etc.) |

After `engine.solve()` finishes, call `engine.report()` to get a
`BranchSessionReport` with branch list, total wall-clock time, pass/fail counts,
final success flag, and the final `TrajectoryStatus`.

```python
from reasonflow.branch_and_share import BranchAndShareConfig, BranchAndShareEngine
from reasonflow.branch_and_share.branch_manager import GitWorktreeBranchManager

config = BranchAndShareConfig(log_repo_root=".")
manager = GitWorktreeBranchManager(".")
engine = BranchAndShareEngine(config, manager, runner_factory=...)
result = engine.solve()
print(result)
print(engine.report())
```

The logger validates that the resolved log path stays inside the configured
repo root and degrades to a no-op if the directory cannot be created.

---

## Files used to produce this document

- `src/reasonflow/branch_and_share/protocol.py`
- `src/reasonflow/branch_and_share/adapter.py`
- `src/reasonflow/branch_and_share/engine.py`
- `src/reasonflow/branch_and_share/control.py`
- `src/reasonflow/branch_and_share/results.py`
- `src/reasonflow/branch_and_share/config.py`
- `src/reasonflow/branch_and_share/detector.py`
- `src/reasonflow/branch_and_share/store.py`
- `src/reasonflow/branch_and_share/packet.py`
- `src/reasonflow/branch_and_share/branch_manager.py`
- `src/reasonflow/branch_and_share/launcher.py`
- `src/reasonflow/branch_and_share/logging.py`
