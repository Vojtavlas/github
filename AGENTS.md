# ReasonFlow project notes

> This file is the single source of truth for how to work in this repo.  
> If you change the repo structure, dependencies, commands, conventions, or domain model, update this file to match before declaring the work complete.

## Quick start

- **Python interpreter:** Use system Python 3.11. The default shell `python`/`pytest` points to a Hermes venv without `torch`.
- **Install:** `py -3.11 -m pip install -e .` (or `py -3.11 -m pip install -e ".[dev]"` for dev tools).
- **Run a demo:** `py -3.11 examples/simple_demo.py --max-new-tokens 10`
- **Benchmark:** `py -3.11 examples/benchmark_demo.py --max-new-tokens 32`
- **Lint:** `ruff check src tests examples`
- **Test (skip engine integration):**
  - PowerShell: `$env:SKIP_ENGINE_TESTS = "1"; py -3.11 -m pytest -q`
  - POSIX: `SKIP_ENGINE_TESTS=1 py -3.11 -m pytest -q`
- **Test (with engine integration):** `py -3.11 -m pytest -q`  
  Engine integration tests load `Qwen/Qwen3.5-0.8B` from Hugging Face Hub.
- **HF token:** Set `HF_TOKEN` for higher Hugging Face Hub rate limits.

## Project overview

**ReasonFlow** (`reasonflow`, v0.1.0) is a small, reproducible research package for fast multi-branch LLM reasoning. It implements RKSC-style inference optimizations:

1. **Prefix KV sharing** — compute the shared problem prompt once and reuse it for every reasoning branch.
2. **ASKS hidden-state similarity gating** — decide when a branch is semantically close enough to the root to reuse its KV cache.
3. **CGEE (Confidence-Gated Early Exit)** — skip or early-exit the verification forward pass when the model is already confident.

The public API lives in `src/reasonflow/__init__.py` and the main entry point is `MultiBranchEngine.solve()`.

## AGENTS.md hierarchy

This root file is the single source of truth for cross-repo conventions. Subsystem details live in child AGENTS.md files:

- `src/reasonflow/AGENTS.md` — core RKSC inference engine modules.
- `src/reasonflow/branch_and_share/AGENTS.md` — Pi coding-agent branching and sharing layer.
- `tests/AGENTS.md` — test conventions and coverage map.

## Domain model

| Component | Responsibility |
|-----------|--------------|
| `Problem` | Input math/word problem string. |
| `Shared prefix` | `"You are a helpful assistant...\n\nProblem: {problem}\n\nReasoning:"` computed once per solve. |
| `Branch hint` | One of `DEFAULT_BRANCH_HINTS` appended to the prefix to encourage a different reasoning strategy per branch. |
| `ASKSManager` (`asks.py`) | Captures root hidden states and KV cache; delegates similarity computation to a `SimilarityMetric` and layer weighting to a `WeightingStrategy`; gates whether a branch can reuse the root KV cache (`score_branch`). |
| `SimilarityMetric` / `WeightingStrategy` (`asks.py`) | Pluggable seam behind ASKS: `SimilarityMetric` computes per-layer or scalar similarity; `WeightingStrategy` combines layer scores (exponential or uniform). |
| `BranchGenerator` (`branch_generator.py`) | Builds a branch prompt, runs prefix+hint prefill, checks ASKS, and decodes directly from the prefill output (no re-tokenization fallback). Patches the torch linear-attention fallback chunk size for short suffixes. `generate_baseline_branch` generates from scratch (no KV reuse). |
| `Decoder` + `Sampler` (`decoder.py`, `sampler.py`) | Autoregressive decoding with temperature, top-p, or greedy; returns token sequence and mean generation confidence. |
| `CGEEAnalyzer` (`cgee.py`) | Two-level confidence gating. Level 1: skip verification if one branch is confident and leads by a wide gap. Level 2: exit the verifier forward pass early when per-layer output entropy becomes low and stable. Composes `EntropyTracker`, `EarlyExitStrategy`, and `HookAdapter`. |
| `Verifier` (`verifier.py`) | Prompts the model with `"Is this answer correct? Answer YES or NO."` and scores the answer by the probability of `YES`. |
| `RSBCMManager` (`cache.py`) | Reasoning-Selective Block Cache Manager: allocates and evicts KV blocks by `importance / (tree_depth + 1)` priority for deep tree searches. |
| `CacheAdapter` (`cache_adapter.py`) | Pluggable adapters for cloning and expanding different Hugging Face KV cache formats (`DynamicCache`, model-specific subclasses, iterable caches, legacy tuples). |
| `ModelAdapter` (`model_adapter.py`) | Registry that locates transformer decoder layers for LLaMA, GPT-2, GPT-NeoX, and heuristic fallback. |
| `MultiBranchEngine` (`engine.py`) | Orchestrates the full RKSC solve: prefix forward, branch generation, CGEE gating, verification, and result aggregation. Also provides `baseline_solve()` for comparison. |
| `EngineConfig` / `RKSCConfig` / `RSBCMConfig` (`config.py`) | Top-level hyperparameters (branching factor, max new tokens, ASKS threshold `tau`, CGEE entropy thresholds, RSBCM capacity). |
| `BranchResult` / `SolveResult` (`results.py`) | Dataclasses for generated text, confidence, verification score, early-exit layer, and timing. |
| `Metrics` (`metrics.py`) | Simple `speedup` and `mean_speedup` helpers for benchmark scripts. |
| `Evaluator` / `EvalConfig` / `EvalReport` (`eval.py`) | Dataset evaluation harness: answer extraction, pluggable metrics (`exact_match`, `numeric_match`, `contains`), and accuracy/speedup reporting. |
| `TrajectoryMonitor` / `StagnationDetector` / `ExperiencePacket` (`branch_and_share/`) | Failure-aware Pi coding-agent layer: `TrajectoryMonitor` records tool calls, repo changes, tests, and tokens; `StagnationDetector` signals repeated commands/file reads, no test improvement, tool failures, code churn, and token-limit approach; `ExperiencePacketBuilder` grounds the summary in git, logs, and test output. |
| `BranchAndShareEngine` (`branch_and_share/engine.py`) | Orchestrates running a Pi trajectory, branching on stagnation, sharing an experience packet, and resuming from the original state or a checkpoint. Hardened to return a `ShareResult`, writes a JSONL session log, and exposes `report() -> BranchSessionReport`. |
| `BranchSessionLogger` (`branch_and_share/logging.py`) | Writes JSON-lines session logs to `<repo_root>/.reasonflow/sessions/<iso-timestamp>.jsonl`; validates path stays inside repo root and degrades to no-op when disabled. |
| `BranchSessionReport` (`branch_and_share/results.py`) | Summary report from `BranchAndShareEngine.report()` with branches, timing, pass/fail counts, and final outcome. |
| `BranchManager` / `GitWorktreeBranchManager` / `MemoryBranchManager` (`branch_and_share/branch_manager.py`) | Creates isolated branches via git worktrees or in-memory references; raises `BranchManagerError` (with command/rc/stderr) for git and validation failures. |
| `PiAdapter` / `MockPiAdapter` / `FileStreamPiAdapter` / `SubprocessPiAdapter` / `TailPiAdapter` (`branch_and_share/adapter.py`) | Plugin seam for the Pi agent loop. `FileStreamPiAdapter` replays a newline-delimited JSON event file; `SubprocessPiAdapter` runs a command and replays its stdout events; `TailPiAdapter` tails a growing JSONL log; `MockPiAdapter` replays scripted scenarios for tests. |
| `ExperienceStore` (`branch_and_share/store.py`) | Append-only JSONL persistence for `ExperiencePacket`s. Skips corrupted lines with a warning, caps loaded history to `max_history`, and loads recent packets for seeding the next branch. |
| `BranchSessionLauncher` (`branch_and_share/launcher.py`) | Creates a branch, seeds `.branch_context.json`, runs the trajectory, and returns `(BranchContext, TrajectoryOutcome, TrajectoryControl)`. |
| `Event` dataclasses / `EventStreamReader` (`branch_and_share/protocol.py`, `branch_and_share/stream.py`) | Typed `Event` subclasses and `_validate_event` centralize JSON parsing; `EventStreamReader` streams chunks into whole lines (CRLF, partial lines, missing trailing newline) with line numbers. |
| `BranchAndShareConfig` / `StagnationConfig` (`branch_and_share/config.py`) | Top-level branch-and-share hyperparameters, including `max_branches`, `max_history`, `timeout_seconds`, `heartbeat_interval`, and `log_repo_root`. |

## Project structure

```
src/reasonflow/
  __init__.py              Public exports; imports MultiBranchEngine only if adapters are present
  config.py                EngineConfig, RKSCConfig, RSBCMConfig
  branch_and_share/        Failure-aware branching layer for Pi coding agents
    __init__.py              Public exports
    config.py                BranchAndShareConfig, StagnationConfig
    results.py               Dataclasses: BranchContext, ExperiencePacket, ShareResult, BranchMetrics, ...
    monitor.py               TrajectoryMonitor: tool/test/file/token history
    detector.py              StagnationDetector and stagnation signals
    packet.py                ExperiencePacketBuilder grounded in git/logs/tests
    protocol.py              JSON event dataclasses and `_validate_event`
    stream.py                `EventStreamReader`: line buffering, split chunks, CRLF, line numbers
    branch_manager.py        BranchManager, GitWorktreeBranchManager, MemoryBranchManager
    metrics.py               MetricsTracker
    adapter.py               PiAdapter, MockPiAdapter, FileStreamPiAdapter, SubprocessPiAdapter
    launcher.py              BranchSessionLauncher
    store.py                 ExperienceStore
    control.py               TrajectoryControl passed to runners (optionally logs actions to BranchSessionLogger)
    engine.py                BranchAndShareEngine orchestration
    logging.py               BranchSessionLogger
  utils.py                 load_model_and_tokenizer, squeeze_hidden
  asks.py                  ASKSManager and pluggable SimilarityMetric / WeightingStrategy
  cgee.py                  CGEEAnalyzer, EntropyTracker, EarlyExitStrategy, HookAdapter
  cache.py                 RSBCMManager block cache
  cache_adapter.py         CacheAdapter subclasses + clone_kv_cache / expand_kv
  model_adapter.py         ModelAdapter registry + get_transformer_layers
  engine.py                MultiBranchEngine orchestration
  decoder.py               Autoregressive Decoder
  sampler.py               Temperature / top-p / greedy Sampler
  verifier.py              CGEE-gated Verifier
  branch_generator.py      Branch generation with ASKS gating
  results.py               BranchResult, SolveResult dataclasses
  metrics.py               Speedup helpers
  eval.py                  Dataset evaluation harness

docs/
  branch_and_share.md    JSON event protocol, custom PiAdapter guide, branching/seeding, and logging/reporting

examples/
  simple_demo.py           Single-problem RKSC vs baseline demo
  benchmark_demo.py        Multi-problem benchmark
  eval_demo.py             Dataset accuracy/speedup evaluation demo
  branch_and_share_demo.py End-to-end branch_and_share demo with git worktrees and a subprocess agent
  pi_agent_stream.py       Streaming demo that tails a live JSONL event log

tests/
  test_asks.py
  test_branch_generator.py
  test_cache.py
  test_cache_adapter.py
  test_cgee.py
  test_config.py
  test_decoder.py
  test_engine.py
  test_model_adapter.py
  test_sampler.py
  test_verifier.py
  test_branch_and_share.py
  test_branch_and_share_integration.py  Integration tests for adapters, launcher, store, and engine
  test_branch_and_share_stream.py      EventStreamReader / protocol unit tests
  test_branch_and_share_subprocess.py  `SubprocessPiAdapter` unit tests
  test_branch_and_share_tail.py        `TailPiAdapter` unit tests
  test_branch_and_share_adversarial.py    Adapter adversarial tests (FileStream / Subprocess)
  test_branch_and_share_store_adversarial.py  `ExperienceStore` corruption / max_history tests
  test_branch_and_share_branch_manager_adversarial.py  `BranchManagerError` and failure-mode tests
  test_branch_and_share_engine_adversarial.py  Engine / launcher failure-mode tests
  test_branch_and_share_logging.py       `BranchSessionLogger`, `engine.report()`, and session JSONL tests

.github/workflows/ci.yml   GitHub Actions CI
pyproject.toml             Package metadata, dependencies, tool configs
```

## Tooling and conventions

- **Build:** `setuptools`, PEP 621 `pyproject.toml`, `src/` layout.
- **Python:** `>=3.9`.
- **Dependencies:** `torch>=2.0`, `transformers>=4.40`, `datasets>=2.14`, `tqdm`, `numpy`, `scipy`.
- **Dev dependencies:** `pytest>=7.0`, `black>=23.0`, `ruff>=0.1.0`, `mypy>=1.0`.
- **Lint/format:** `ruff` and `black` use `line-length = 100` and `target-version = "py39"`. Ruff selects `E`, `F`, `W`, `I`.
- **Type check:** `mypy` targets `python_version = "3.11"`.
- **License:** MIT.
- **Commit style:** Recent history uses Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`). Prefer that style.

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`:

- Python 3.10 and 3.11 matrix.
- `pip install -e ".[dev]"`
- `ruff check src tests examples`
- `pytest -q` with `SKIP_ENGINE_TESTS=1`

## Branches and worktrees

- **Main branch:** `main`
- **Remote:** `origin -> https://github.com/Vojtavlas/ReasonFLow`
- **Existing worktrees** (`.worktrees/` is in `.gitignore`):
  - `.worktrees/asks-metric` on branch `rf/deep-asks`
  - `.worktrees/cache-adapter` on branch `rf/deep-cache`
  - `.worktrees/cgee-hooks` on branch `rf/deep-cgee`
  - `.worktrees/engine-orchestration` on branch `rf/deep-engine`
  - `.worktrees/model-adapter` on branch `rf/deep-model`

Use `ce-worktree` (preferred) or `using-git-worktrees` when starting isolated feature work. Always keep `.worktrees/` gitignored.

## Testing conventions

- `pytest` with tests named `test_*.py`.
- Engine integration tests in `test_engine.py` load `Qwen/Qwen3.5-0.8B` and are skipped when `SKIP_ENGINE_TESTS=1`.
- Prefer real code over mocks unless unavoidable.
The current baseline is: `ruff check src tests examples` clean; `SKIP_ENGINE_TESTS=1 py -3.11 -m pytest -q` reports `255 passed, 4 skipped`; a full run `py -3.11 -m pytest -q` reports `254 passed, 5 skipped`.

## Benchmark conventions

- Run: `py -3.11 examples/benchmark_demo.py --max-new-tokens 32`.
- Use `--warmup 1` and `--runs 3` (or more) and alternate RKSC/baseline order to avoid GPU/cache-order bias.
- `torch.cuda.synchronize()` is used around each timed call for wall-clock GPU measurements.
- Default to greedy (`--temperature 0.0`) for stable, reproducible speed comparisons.
- Report mean speedup across the problem set, not a single prompt.
- Current reference result (RTX 3050, Qwen 0.8B, `max_new_tokens=32`, 3 runs, 1 warmup):
  - Mean RKSC: 3042.3 ms
  - Mean baseline: 3635.7 ms
  - Speedup: 1.20x
- Current measurement (local CUDA, Qwen 0.8B, `max_new_tokens=32`, 3 runs, 1 warmup):
  - Mean RKSC: 3662.1 ms
  - Mean baseline: 3618.7 ms
  - Speedup: 0.99x

## Skill map: what to use and when

At the start of every task, invoke `using-superpowers` to surface applicable skills. This repo prefers the Compound Engineering (`ce-*`) workflow because it is the most complete for this codebase. Default to the skills below; only invoke a skill not on this list if the task clearly falls outside these categories.

| Skill | When to use | Notes |
|-------|-------------|-------|
| `using-superpowers` | Every task start | Identifies applicable skills; this `AGENTS.md` takes precedence over generic skill defaults. |
| `verification-before-completion` | Before claiming any task is done/fixed/passing | Run `ruff check src tests examples` and `py -3.11 -m pytest -q` (with or without `SKIP_ENGINE_TESTS=1`) and read the output. Evidence before claims. |
| `ce-brainstorm` | Scoping a new feature, behavior change, or ambiguous idea | Produces a `docs/plans/` requirements-only unified plan. Use before `ce-plan`. |
| `ce-plan` | Turning requirements/spec into an implementation plan | Enriches a `ce-brainstorm` artifact or works from a clear prompt. |
| `ce-worktree` | Starting isolated feature work or reviewing a branch/PR in isolation | Creates/attaches worktrees under `.worktrees/`, keeps `main` clean. |
| `ce-work` | Implementing a plan or a concrete build/change prompt end-to-end | Handles task breakdown, subagents, and local verification. |
| `dispatching-parallel-agents` | Ad-hoc parallel dispatch of 2+ independent exploration/debug/review tasks | Use when the work is not part of a formal `ce-work` plan (see Subagent workflows). |
| `test-driven-development` | Writing new behavior or fixing a bug | Write a failing test first, then the minimal code, then verify. |
| `ce-debug` | Investigating test failures, regressions, stack traces, or unexpected behavior | Use before proposing fixes. |
| `ce-code-review` | Reviewing a batch of changes before commit/PR | Multi-persona review; default applies safe fixes. |
| `ce-commit` | Committing staged/unstaged changes | Use when the user asks to commit. |
| `ce-commit-push-pr` | Shipping / opening a PR | Use when the user asks to push and open a PR. |
| `ce-resolve-pr-feedback` | Addressing PR review comments | Use after a PR has feedback. |
| `ce-simplify-code` | Refactor/tidy pass on recently changed code | Preserves behavior; use after a feature lands. |
| `ce-compound` | Recording a durable learning after solving a non-trivial problem | Write to `docs/solutions/` or `CONCEPTS.md`. |

### Installed skills intentionally not in the core map

These overlap with the core map above or are for other kinds of work:

- `brainstorming`, `writing-plans`, `executing-plans`, `subagent-driven-development` — overlap with `ce-brainstorm` / `ce-plan` / `ce-work`.
- `systematic-debugging` — overlap with `ce-debug`.
- `using-git-worktrees` — overlap with `ce-worktree` (`ce-worktree` is preferred because it integrates with the rest of the `ce-*` pipeline).
- `requesting-code-review`, `receiving-code-review` — overlap with `ce-code-review` / `ce-resolve-pr-feedback`.
- `finishing-a-development-branch` — overlap with `ce-commit-push-pr` and `ce-worktree` cleanup.
- `ce-babysit-pr`, `ce-test-browser`, `computer-use`, `orca-cli`, `orchestration`, `ce-riffrec-feedback-analysis`, `ce-pov`, `ce-ideate`, `ce-strategy`, `ce-explain`, `ce-optimize`, `ce-proof`, `ce-handoff`, `ce-doc-review` — not applicable to routine work in this Python research repo.

If a future task clearly needs one of these, use it; otherwise default to the core map.

## Subagent-first execution workflows

The codebase is small enough to read quickly, but it is intentionally modular (ASKS, CGEE, cache adapters, model adapters, engine, sampler/decoder, verifier). **Default to subagents** for any task that touches more than one file or needs more than a few minutes of exploration. Inline work is reserved for one-liners and trivial fixes.

### When and how many agents

| Work type | When to dispatch | Typical count | Agent / skill | What each agent does |
|-----------|------------------|---------------|---------------|----------------------|
| Broad repo exploration | Starting work in an unfamiliar area; mapping the domain model | 3–5, up to 10 for a full sweep | `subagent_explore` (or `ce-work` scout phase) | Each reads a focused slice (one source file or one test file) and returns a concise summary + dependencies. The main agent synthesizes. |
| Multi-unit implementation | A `ce-plan` has independent implementation units | 1 per unit, batched by dependency layer | `subagent_general` inside `ce-work` | Implement one unit, write focused tests, self-review, and report evidence. |
| Parallel debugging | 2+ unrelated test failures or independent bugs | 1 per failure domain | `subagent_general` (or `ce-debug` for a single deep bug) | Investigate root cause independently, then merge non-conflicting fixes. |
| Cross-cutting review | Large change touches many files or risky areas | `ce-code-review` already spawns a roster | Generic subagents with reviewer personas | Run correctness, testing, maintainability, and performance lenses. |
| Benchmark / tuning | Comparing hyperparameters, samplers, or cache strategies | 3–5 | `subagent_general` | Each runs one configuration; main agent aggregates speed/accuracy results. |
| Documentation / learnings | After solving a non-trivial problem | 1 | `subagent_explore` or main agent with `ce-compound` | Draft `docs/solutions/` or `CONCEPTS.md` entry for future agents. |

### Example: 10-agent repo sweep

When ramping up on a new component or verifying the state of `src/reasonflow/`, dispatch up to 10 `subagent_explore` agents in parallel, one per file:

```
subagent 1 -> read src/reasonflow/asks.py      -> summarize ASKS gating and metrics
subagent 2 -> read src/reasonflow/cgee.py      -> summarize CGEE levels and hooks
subagent 3 -> read src/reasonflow/cache.py     -> summarize RSBCM block cache
...
```

Each agent returns: (1) the file's single responsibility, (2) its main classes/functions, (3) its direct dependencies in this repo, and (4) any obvious risks. The main agent then synthesizes these into a coherent plan.

### Dispatch rules

- **One task per agent.** Broad prompts produce broad, unusable answers.
- **Batch by independence.** Only run agents in parallel when their files/workspaces do not overlap; serialize when they touch the same module or shared state.
- **Cap at ~5 for implementation, ~10 for exploration.** More agents add merge overhead without adding signal.
- **Hand off artifacts as files.** Give each agent a brief file and ask it to write its report to a uniquely named file; do not paste large outputs back into the main context.
- **Use `ce-work` for planned implementation** and `dispatching-parallel-agents` for ad-hoc parallel exploration or debugging (2+ independent tasks outside a formal plan).

### Skill note

`dispatching-parallel-agents` is the recommended skill for ad-hoc parallel dispatch of independent exploration, debugging, or verification tasks. `ce-work` already handles parallel subagents when executing an implementation plan, so do not double-dispatch the same work.

## Anti-patterns

- Do not let `RSBCMManager` temporarily hold more blocks than `max_blocks`; evict before allocating.
- Do not remove the catch-all safety net in `model_adapter.py`; the heuristic adapter is the default fallback.
- Do not assume `MultiBranchEngine` is importable in every worktree; it is guarded by a conditional `try/except`.
- Do not delete failing tests just to make a suite pass.
- Do not double-dispatch the same work through both `ce-work` and `dispatching-parallel-agents`.
- Do not paste large agent outputs back into the main context; hand off artifacts as files.
- Do not claim a task is done/fixed/passing without running the verification commands and reading the output.

## Unique styles

- Two subsystems share one package: `reasonflow` (RKSC LLM inference) and `reasonflow.branch_and_share` (Pi agent branching). They have separate public APIs and separate test suites.
- Agent-oriented docs live at the repo root (`AGENTS.md`, `CONTEXT.md`, `TASKS.md`) and are the authoritative spec; `README.md` is public-facing and can be stale.
- Subagent-first execution is the default for any non-trivial work; inline edits are reserved for one-liners.
- Tests are split flat in `tests/` with dedicated `*_adversarial.py`, `*_integration.py`, `*_stream.py`, `*_subprocess.py`, `*_tail.py`, and `*_logging.py` files for the branch_and_share subsystem.
- `.worktrees/` under `.gitignore` is the preferred isolation mechanism for feature work.

## Notes

- The default `python`/`pytest` in this environment routes to a Hermes venv without `torch`; use `py -3.11` for all dev commands.
- `README.md`'s `Project structure` section is stale (it omits `branch_and_share/`, `cache_adapter.py`, `model_adapter.py`, `verifier.py`, `branch_generator.py`, `decoder.py`, `sampler.py`, `results.py`); trust `AGENTS.md`.
- The `.codegraph/` index is present but inactive in this session; rely on `glob`/`grep`/`read` and the subagent reports.
- `.worktrees/` is documented in AGENTS.md but does not currently exist on disk (worktrees likely pruned or on another branch).
- `.reasonflow/sessions/` is created on demand by `BranchSessionLogger`.
- Engine integration tests download `Qwen/Qwen3.5-0.8B` from Hugging Face Hub; set `HF_TOKEN` for higher rate limits and use `SKIP_ENGINE_TESTS=1` to skip them.

## When to update this file

Update this file whenever you:

- Add, remove, or rename source files, tests, examples, or dependencies.
- Change verification commands, CI, install steps, or environment requirements.
- Modify the domain model, public API, or key class responsibilities.
- Change skill usage conventions.

After any edit, re-read the file against the actual repo state to confirm it is still accurate.
