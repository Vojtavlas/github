# Test suite

## OVERVIEW
`tests/*.py` covers the 31 source modules under `src/reasonflow/`. The suite is flat and splits branch_and_share concerns into unit, integration, adversarial, stream, subprocess, tail, and logging tests.

## WHERE TO LOOK
| Source module | Test file(s) |
|---|---|
| `reasonflow.engine` | `test_engine.py` (HF model integration, `SKIP_ENGINE_TESTS` gated) |
| `reasonflow.asks` | `test_asks.py` |
| `reasonflow.cgee` | `test_cgee.py` |
| `reasonflow.cache` | `test_cache.py` |
| `reasonflow.cache_adapter` | `test_cache_adapter.py` |
| `reasonflow.model_adapter` | `test_model_adapter.py` |
| `reasonflow.branch_generator` | `test_branch_generator.py` |
| `reasonflow.decoder` / `sampler` | `test_decoder.py` / `test_sampler.py` |
| `reasonflow.verifier` | `test_verifier.py` |
| `reasonflow.config` / `results` / `metrics` / `utils` | `test_config.py` / `test_results.py` / `test_metrics.py` / `test_utils.py` |
| `branch_and_share.*` | `test_branch_and_share*.py` |

## CONVENTIONS
- `test_*.py` flat in `tests/`; no `conftest.py`.
- Inline fixtures per file; prefer real code over mocks.
- Use `tmp_path` for JSONL logs, git repos, and `ExperienceStore` files.
- `test_engine.py` loads `Qwen/Qwen3.5-0.8B` and is skipped by `SKIP_ENGINE_TESTS=1`. `test_branch_and_share_integration.py` runs offline.
- Baseline fast suite: `241 passed, 4 skipped`; full run: `240 passed, 5 skipped`.

## ANTI-PATTERNS
- Do not add HF model-dependent tests without a `SKIP_ENGINE_TESTS` guard.
- Do not put branch_and_share tests in a subdirectory; keep `tests/` flat with descriptive suffixes.
- Avoid heavy mocks when a real object or `FakeModel`/`FakeTokenizer` suffices.