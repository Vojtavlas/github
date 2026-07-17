# RKSC inference engine

## OVERVIEW
`src/reasonflow/*.py` implements the core RKSC-style multi-branch LLM reasoning engine: prefix KV sharing, ASKS gating, CGEE early exit, and branch generation/verification.

## WHERE TO LOOK
| Task | Location | Notes |
|---|---|---|
| Entry point | `engine.py` | `MultiBranchEngine.solve()` / `baseline_solve()` |
| Public config | `config.py` | `EngineConfig`, `RKSCConfig`, `RSBCMConfig` |
| Prefix KV sharing + ASKS gate | `asks.py` | `ASKSManager`, `SimilarityMetric`, `WeightingStrategy` |
| Confidence-gated early exit | `cgee.py` | `CGEEAnalyzer`, `EntropyTracker`, `EarlyExitStrategy`, `HookAdapter` |
| Block cache manager | `cache.py` | `RSBCMManager` |
| KV cache format adapters | `cache_adapter.py` | `CacheAdapter` subclasses |
| Model layer lookup | `model_adapter.py` | `ModelAdapter` registry |
| Branch prompt + decode | `branch_generator.py` | `BranchGenerator` |
| Autoregressive decoding | `decoder.py` / `sampler.py` | `Decoder`, `Sampler` |
| Verification scoring | `verifier.py` | `Verifier` |
| Result dataclasses | `results.py` | `BranchResult`, `SolveResult` |
| Dataset evaluation | `eval.py` | `Evaluator`, `EvalConfig`, `EvalReport`, `HFTextDataset`, `AnswerExtractor`, metrics |
| Shared utilities | `utils.py` / `metrics.py` | `load_model_and_tokenizer`, `speedup`, `mean_speedup` |
| Pi agent layer | `branch_and_share/` | See `branch_and_share/AGENTS.md` |

## CODE MAP
| Symbol | Type | Location | Role |
|---|---|---|---|
| `MultiBranchEngine` | class | `engine.py` | End-to-end RKSC solve orchestration |
| `EngineConfig` | dataclass | `config.py` | Top-level hyperparameters |
| `RKSCConfig` | dataclass | `config.py` | ASKS/CGEE thresholds |
| `RSBCMConfig` | dataclass | `config.py` | Cache capacity |
| `ASKSManager` | class | `asks.py` | Hidden-state similarity gate for KV reuse |
| `CGEEAnalyzer` | class | `cgee.py` | Confidence-gated early exit |
| `RSBCMManager` | class | `cache.py` | Score/depth block eviction |
| `BranchGenerator` | class | `branch_generator.py` | Prefix+hint prefill and decode |
| `Decoder` | class | `decoder.py` | Autoregressive loop |
| `Sampler` | class | `sampler.py` | Temperature / top-p / greedy |
| `Verifier` | class | `verifier.py` | YES-probability scoring |
| `ModelAdapter` | class | `model_adapter.py` | Model layer registry |
| `CacheAdapter` | class | `cache_adapter.py` | KV cache clone/expand |
| `load_model_and_tokenizer` | function | `utils.py` | Bootstrap a HF model |
| `BranchResult` / `SolveResult` | dataclass | `results.py` | Generation/verification results |
| `Evaluator` | class | `eval.py` | Accuracy/speedup dataset evaluation harness |

## CONVENTIONS
Same repo-wide: `py -3.11`, ruff/black line length 100, target py39, mypy target 3.11, pytest `test_*.py`. `MultiBranchEngine` is imported conditionally in `__init__.py` because it depends on `cache_adapter`/`model_adapter`, which may be absent in some worktrees.

## ANTI-PATTERNS
- A branch identical to the root always reuses the root KV cache (similarity 1.0).
- Do not use `as any` / `@ts-ignore` style type suppression.
- Do not run engine integration tests in CI; they are gated by `SKIP_ENGINE_TESTS=1`.