# Pi agent branching and sharing layer

## OVERVIEW
`src/reasonflow/branch_and_share/*.py` is a failure-aware branching layer for Pi coding-agent trajectories: it monitors progress, detects stagnation, creates git/memory branches, builds experience packets, and writes JSONL session logs.

## STRUCTURE
```
branch_and_share/
├── adapter.py        # PiAdapter, FileStream/Subprocess/Tail/Mock adapters
├── branch_manager.py # BranchManager, GitWorktree/Memory branch managers
├── config.py         # BranchAndShareConfig, StagnationConfig
├── control.py        # TrajectoryControl
├── detector.py       # StagnationDetector
├── engine.py         # BranchAndShareEngine orchestration
├── launcher.py       # BranchSessionLauncher
├── logging.py        # BranchSessionLogger
├── metrics.py        # MetricsTracker
├── monitor.py        # TrajectoryMonitor
├── packet.py         # ExperiencePacketBuilder
├── protocol.py       # Event dataclasses / JSON validation
├── results.py        # BranchContext, ShareResult, BranchSessionReport, ...
├── store.py          # ExperienceStore (JSONL)
└── stream.py         # EventStreamReader
```

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Entry point | `engine.py` | `BranchAndShareEngine.solve()` returns `ShareResult` |
| Public API | `__init__.py` | Exports 30 symbols across results, managers, adapters, config |
| Branch creation | `branch_manager.py` | `GitWorktreeBranchManager`, `MemoryBranchManager` |
| Pi agent adapters | `adapter.py` | `FileStreamPiAdapter`, `SubprocessPiAdapter`, `TailPiAdapter`, `MockPiAdapter` |
| Stagnation detection | `detector.py` / `monitor.py` | `StagnationDetector`, `TrajectoryMonitor` |
| Experience packets | `packet.py` / `results.py` | `ExperiencePacketBuilder`, `ExperiencePacket`, `ShareResult` |
| Session logging | `logging.py` | Writes `.reasonflow/sessions/<iso>.jsonl` |
| Event protocol | `protocol.py` / `stream.py` | Typed `Event` subclasses, `EventStreamReader` |

## CODE MAP
| Symbol | Type | Location | Role |
|---|---|---|---|
| `BranchAndShareEngine` | class | `engine.py` | Orchestrate Pi trajectory + branches |
| `BranchAndShareConfig` | dataclass | `config.py` | Top-level config |
| `StagnationConfig` | dataclass | `config.py` | Stagnation thresholds |
| `BranchManager` | class | `branch_manager.py` | Abstract branch creation |
| `GitWorktreeBranchManager` | class | `branch_manager.py` | Isolated git-worktree branches |
| `MemoryBranchManager` | class | `branch_manager.py` | In-memory branch refs |
| `PiAdapter` | class | `adapter.py` | Base adapter for Pi events |
| `FileStreamPiAdapter` | class | `adapter.py` | Replay JSONL event file |
| `SubprocessPiAdapter` | class | `adapter.py` | Run command and replay stdout |
| `TailPiAdapter` | class | `adapter.py` | Tail a growing JSONL log |
| `TrajectoryMonitor` | class | `monitor.py` | Record tool calls/tests/files/tokens |
| `StagnationDetector` | class | `detector.py` | Signal repeated commands, no test improvement, etc. |
| `ExperiencePacketBuilder` | class | `packet.py` | Ground summary in git/logs/tests |
| `ExperienceStore` | class | `store.py` | Append-only JSONL history |
| `BranchSessionLogger` | class | `logging.py` | Write JSONL session logs |
| `BranchSessionReport` | dataclass | `results.py` | Final report from `engine.report()` |

## CONVENTIONS
Same repo conventions (ruff/black line length 100, py39, mypy 3.11). Test files use dedicated `*_adversarial.py`, `*_integration.py`, `*_stream.py`, `*_subprocess.py`, `*_tail.py`, `*_logging.py` suffixes. Prefer real code over mocks; use `tmp_path` for filesystem fixtures. No `conftest.py`.

## ANTI-PATTERNS
- `ExperienceStore` warns (not errors) on corrupted JSONL lines; do not change this to hard fail without a plan.
- `BranchManagerError` must carry `command`, `rc`, and `stderr` for git failures.
- Session log paths must stay inside the repo root.