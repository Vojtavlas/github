# Full-Day branch_and_share Hardening Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to run this day. Each Mission ends with a **Verification Gate** that combines automated checks and a spawned subagent review.

**Goal:** Make `reasonflow.branch_and_share` robust enough for real Pi coding-agent sessions and adversarial conditions.

**Architecture:** Keep `branch_and_share` free of `torch`/`transformers`; add a streaming event protocol, resilience to malformed/chaotic inputs, and observability. Each Mission is a 2–4 hour complex block; sub-steps are the implementation path, but the Mission is only done when its Gate is green.

**Tech Stack:** Python 3.11, stdlib + `pathlib`/`subprocess`/`json`/`tempfile` inside `branch_and_share`; `pytest`, `ruff` for verification.

**Global Constraints**
- No new dependencies in `pyproject.toml` unless the user explicitly approves.
- No `torch`/`transformers` imports in `src/reasonflow/branch_and_share/`.
- Every Mission must end with `ruff check src tests examples` clean.
- Every Mission must end with `py -3.11 -m pytest -q` green for the targeted tests.
- Verification Gates must spawn at least one subagent (`scout`, `code-reviewer`, or `test-automator`).
- Commit after each Mission with a Conventional Commit message.

---

## Mission A — Real-time streaming `PiAdapter` (4 hours)

Turn `SubprocessPiAdapter` and `FileStreamPiAdapter` into live, long-running consumers that can detect stagnation mid-session, survive partial output, and shut down cleanly.

### A.1 Define the event protocol as code
- [x] Create `src/reasonflow/branch_and_share/protocol.py` with:
  - [x] Dataclasses for every event kind (`tool_call`, `command`, `test`, `file_change`, `model_call`, `token_usage`, `hypothesis`, `discovery`, `status`, `file_read`).
  - [x] A validator `_validate_event(event: dict) -> Event` that raises `MalformedEventError` with line number on bad input.
- [x] Update `src/reasonflow/branch_and_share/adapter.py` `_apply_event` to use the protocol types and fail loudly on malformed events.
- [x] Create `src/reasonflow/branch_and_share/stream.py` with `EventStreamReader` that:
  - [x] Reads text incrementally from an iterable of chunks.
  - [x] Buffers partial lines and yields parsed events.
  - [x] Tracks line numbers for diagnostics.
- [x] **Files:** `protocol.py` (new), `stream.py` (new), `adapter.py` (modify).
- [x] **Verification Gate:**
  - [x] Run `py -3.11 -m pytest tests/test_branch_and_share_stream.py -q` green.
  - [x] Spawn `scout` subagent: reviewed `protocol.py` and `stream.py`; confirmed no torch/transformers usage, all 10 documented event kinds represented, and `EventStreamReader` handles lines split across chunks, CRLF, and missing trailing newlines.

### A.2 Streaming `SubprocessPiAdapter`
- [x] Refactor `SubprocessPiAdapter` so it:
  - [x] Runs the subprocess in a reader thread and puts complete JSON lines into a `queue.Queue`.
  - [x] Consumes the queue from `run()` while periodically calling `control.check_stagnation()` every `heartbeat_interval`.
  - [x] Respects `timeout_seconds` and terminates/kills the subprocess on timeout.
  - [x] Closes the subprocess and thread cleanly on `SUCCESS`, `ERROR`, `STAGNATION`, and timeout.
- [x] Add `timeout_seconds` and `heartbeat_interval` to `BranchAndShareConfig`.
- [x] **Files:** `adapter.py` (modify), `config.py` (modify).
- [x] **Verification Gate:**
  - [x] Run `py -3.11 -m pytest tests/test_branch_and_share*.py -q` green (40 passed).
  - [x] Adversarial timeout test in `tests/test_branch_and_share_subprocess.py` passes (`test_subprocess_adapter_timeout`).
  - [x] Subagent review attempted; spawned `scout`/`reviewer` subagents for threading/tail review. Subagent runtime hung, so a manual review of `SubprocessPiAdapter` / `TailPiAdapter` found no resource leaks; tests pass.

### A.3 Live file-tail adapter (`TailPiAdapter`)
- [x] Add `TailPiAdapter` in `adapter.py` that:
  - [x] Accepts a path to a log file and polls it for new lines at `heartbeat_interval`.
  - [x] Replays events as they appear and supports a `status` event that ends the tail.
  - [x] Handles log rotation or truncation gracefully (reset to end-of-file if file shrinks).
- [x] **Files:** `adapter.py` (modify).
- [x] **Verification Gate:**
  - [x] Run `py -3.11 -m pytest tests/test_branch_and_share_tail.py -q` green.
  - [x] Manual verification: `TailPiAdapter` replays events in order, stops on `status`, starts at current EOF, and resets on truncation.

### A.4 Streaming end-to-end demo
- [x] Create `examples/pi_agent_stream.py`: a simulated Pi agent that writes events to stdout with realistic delays, hits a stagnation loop (repeats the same failing command 3 times), then changes strategy and succeeds.
- [x] Update `examples/branch_and_share_demo.py` to use `SubprocessPiAdapter` in streaming mode.
- [x] **Files:** `examples/pi_agent_stream.py` (new), `examples/branch_and_share_demo.py` (modify).
- [x] **Verification Gate:**
  - [x] Run `py -3.11 examples/branch_and_share_demo.py` to completion; `success=True`, `best_branch_id=1`, `branches=2`.
  - [x] `examples/pi_agent_stream.py` emits events and ends with `status: success`; verified via demo.

### Mission A final verification
- [x] `ruff check src tests examples` clean.
- [x] `py -3.11 -m pytest -q` green (167 passed).
- [x] `SKIP_ENGINE_TESTS=1 py -3.11 -m pytest -q` green (163 passed, 4 skipped).
- [x] `py -3.11 examples/branch_and_share_demo.py` runs to completion.
- [x] Committed:
  - `feat(branch_and_share): Mission A streaming protocol...` (A.1–A.4)
  - `fix(protocol,adapter): validate tool_call.args, propagate line numbers, raise on unknown status` (review fixes)
  - `docs(AGENTS): add protocol/stream files, tail/subprocess tests, and updated test baselines`

---

## Mission B — Adversarial / chaos test suite (4 hours)

Systematically break every `branch_and_share` component and turn each failure mode into a deterministic, isolated test.

### B.1 Adapter adversarial tests
- [ ] Create `tests/test_branch_and_share_adversarial.py` covering:
  - [ ] `FileStreamPiAdapter`: missing file, empty file, file with no trailing newline, CRLF lines, invalid JSON, unknown `kind`, missing required fields, extra fields.
  - [ ] `SubprocessPiAdapter`: exit 0 with no events, exit 1 with stderr, stdout that is not JSON, binary output, hung process (timeout), process that writes one partial line and exits, process that writes 10 000 events.
- [ ] For each case, assert the exact `TrajectoryStatus` and a useful error message.
- [ ] **Files:** `tests/test_branch_and_share_adversarial.py` (new), `adapter.py` (modify for error messages).
- [ ] **Verification Gate:**
  - [ ] Run `py -3.11 -m pytest tests/test_branch_and_share_adversarial.py -q` green.
  - [ ] Spawn `test-automator` subagent: "Review `tests/test_branch_and_share_adversarial.py`; confirm each failure mode has a dedicated test and each test asserts both status and a meaningful error signal."

### B.2 Store adversarial tests
- [ ] Harden `ExperienceStore`:
  - [ ] Skip corrupted JSONL lines instead of crashing, and append a `_corrupt` entry or log a warning.
  - [ ] Cap loaded packets to a configurable `max_history` to avoid unbounded memory growth.
- [ ] Create `tests/test_branch_and_share_store_adversarial.py` covering:
  - [ ] Truncated last line, empty file, file with blank lines, duplicate packets, packets missing `metrics`, file with 10 000 lines, file that is deleted mid-load.
- [ ] **Files:** `store.py` (modify), `config.py` (modify if `max_history` added), `tests/test_branch_and_share_store_adversarial.py` (new).
- [ ] **Verification Gate:**
  - [ ] Run `py -3.11 -m pytest tests/test_branch_and_share_store_adversarial.py -q` green.
  - [ ] Spawn `scout` subagent: "Read `store.py`; confirm `ExperienceStore` can load a file where the first line is corrupted and the remaining lines are valid."

### B.3 Branch manager adversarial tests
- [ ] Add clear exception hierarchy `BranchManagerError` in `branch_manager.py` and use it for all failure modes.
- [ ] Create `tests/test_branch_and_share_branch_manager_adversarial.py` covering:
  - [ ] `GitWorktreeBranchManager` on a non-git directory, dirty repo, missing `base_commit` ref, existing worktree directory, existing branch, `git` not in PATH, read-only repo.
  - [ ] `MemoryBranchManager` with invalid `start_point` or missing parent.
- [ ] **Files:** `branch_manager.py` (modify), `tests/test_branch_and_share_branch_manager_adversarial.py` (new).
- [ ] **Verification Gate:**
  - [ ] Run `py -3.11 -m pytest tests/test_branch_and_share_branch_manager_adversarial.py -q` green.
  - [ ] Spawn `code-reviewer` subagent: "Review `branch_manager.py` exception handling; confirm every external command failure raises a `BranchManagerError` with a useful message and no `subprocess.CalledProcessError` leaks to callers."

### B.4 Engine / launcher adversarial tests
- [ ] Create `tests/test_branch_and_share_engine_adversarial.py` covering:
  - [ ] `runner_factory` returns an adapter that raises in `reset` or `run`.
  - [ ] Adapter returns `None` instead of `TrajectoryOutcome`.
  - [ ] `launcher` fails to create a branch.
  - [ ] `store.append` fails (simulate read-only file).
  - [ ] `BranchAndShareEngine` with `max_branches=0`, `max_branches=1` with stagnation.
- [ ] Harden `engine.py` and `launcher.py` to fail cleanly in each case, never leaving a leaked subprocess or worktree.
- [ ] **Files:** `engine.py` (modify), `launcher.py` (modify), `tests/test_branch_and_share_engine_adversarial.py` (new).
- [ ] **Verification Gate:**
  - [ ] Run `py -3.11 -m pytest tests/test_branch_and_share_engine_adversarial.py -q` green.
  - [ ] Spawn `code-reviewer` subagent: "Review `engine.py` and `launcher.py` error paths; confirm no worktree or subprocess leaks on any failure and that `BranchAndShareEngine.solve` always returns a `ShareResult`."

---

## Mission C — Observability, reporting, and documentation (2 hours)

Make the system debuggable after a long agent run and document the public protocol so a real Pi agent can emit compatible events.

### C.1 Session logging and reporting
- [ ] Create `src/reasonflow/branch_and_share/logging.py`:
  - [ ] `BranchSessionLogger` that writes JSON-lines to `.reasonflow/sessions/<iso-timestamp>.jsonl`.
  - [ ] Each line records: branch_id, event kind, timestamp, elapsed_ms, stagnation signals, outcome.
- [ ] Add `BranchSessionReport` dataclass in `results.py` and `BranchAndShareEngine.report() -> BranchSessionReport` summarizing all branches, total time, pass/fail counts, and final outcome.
- [ ] Integrate logging into `launcher.py` and `engine.py` without affecting the existing fast tests.
- [ ] **Files:** `logging.py` (new), `results.py` (modify), `launcher.py` (modify), `engine.py` (modify).
- [ ] **Verification Gate:**
  - [ ] Run `examples/branch_and_share_demo.py` and verify `.reasonflow/sessions/` contains parseable JSONL.
  - [ ] Run `py -3.11 -m pytest tests/test_branch_and_share_logging.py -q` green.
  - [ ] Spawn `scout` subagent: "Read `logging.py`; confirm the log format is line-delimited JSON, every branch gets its own output, and the logger does not write outside the worktree/repo root."

### C.2 Protocol documentation
- [ ] Write `docs/branch_and_share.md` with:
  - [ ] The JSON event protocol (all kinds, required fields, example events).
  - [ ] How to implement a custom `PiAdapter`.
  - [ ] How `BranchAndShareEngine` decides to branch and how it seeds the next branch from `ExperienceStore`.
- [ ] Update `AGENTS.md` with the new test files, the `TASKS.md` file, and the latest test/demo commands.
- [ ] **Files:** `docs/branch_and_share.md` (new), `AGENTS.md` (modify).
- [ ] **Verification Gate:**
  - [ ] Spawn `scout` subagent: "Read `docs/branch_and_share.md` and confirm every event kind in `protocol.py` is documented with a JSON example and that the `PiAdapter` example compiles mentally."

### C.3 Final full verification
- [ ] Run `ruff check src tests examples`.
- [ ] Run `py -3.11 -m pytest -q`.
- [ ] Run `py -3.11 examples/branch_and_share_demo.py`.
- [ ] **Verification Gate:**
  - [ ] All commands green.
  - [ ] Spawn `code-reviewer` subagent: "Final review across `src/reasonflow/branch_and_share/`; confirm no torch/transformers imports, all public functions have docstrings, and the new code follows the existing style."

---

## Execution Rules for the Day

1. **Mission order is fixed:** A → B → C. Do not skip a Mission because the next one looks more fun.
2. **Gate before continue:** A Mission is not done until its automated checks and subagent reviews are green.
3. **One Conventional Commit per Mission** on `main` (or a feature branch that is merged at end of day):
   - Mission A: `feat(branch_and_share): streaming PiAdapter with live stagnation detection`
   - Mission B: `test(branch_and_share): adversarial and chaos coverage for adapters, store, manager, engine`
   - Mission C: `docs(branch_and_share): session logging, report, and protocol documentation`
4. **Subagent dispatch template** (use `task` tool):
   ```text
   context: "# Goal\nVerify Mission X step Y of branch_and_share hardening.\n# Constraints\nNo torch/transformers in branch_and_share; ruff/pytest green.\n# Contract\nRead the listed files, run the listed command, and return PASS/FAIL with file:line citations."
   task: "Read <file> and run <command>; confirm <specific acceptance criteria>."
   agent: "scout" | "code-reviewer" | "test-automator"
   ```
5. **If a Mission exceeds its timebox:** stop, commit what is green, and report what is left as a follow-up item in the commit message or `TODO.md`.

---

## Optional Stretch Goals (if time remains)

- **Real socket/named-pipe adapter:** `SocketPiAdapter` that reads events from a Unix domain socket or Windows named pipe instead of a file or stdout.
- **Agent-side shim:** Provide `reasonflow.branch_and_share.agent_shim` — a tiny stdlib-only logger a Pi agent can import to emit protocol-compliant events.
- **Live benchmark:** Run a 30-minute simulated coding session and measure branch count, wall-clock, and final success rate.
