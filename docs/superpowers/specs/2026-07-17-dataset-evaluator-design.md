# ReasonFlow Dataset Evaluator — Design Spec

> Date: 2026-07-17
> Author: Sisyphus / ReasonFlow agent
> Status: design draft

## 1. Problem

`ReasonFlow` currently has `examples/benchmark_demo.py`, which measures wall-clock speedup of `MultiBranchEngine.solve()` against `baseline_solve()`. It does **not** measure whether the generated answers are correct. A research package that claims to accelerate multi-branch reasoning needs a reproducible way to report the accuracy/speedup trade-off. Without it, speed improvements are meaningless if correctness drops.

## 2. Goal

Add a `reasonflow.eval` subsystem that can:

1. Load any Hugging Face `datasets` reasoning dataset (GSM8K, MATH, etc.).
2. Run `MultiBranchEngine` (RKSC) and `baseline_solve` (no sharing) over a configurable subset.
3. Extract a clean final answer from generated text for each problem.
4. Compare extracted answers to gold answers with pluggable metrics (`exact_match`, `numeric_match`, `contains`).
5. Aggregate accuracy, mean speedup, per-problem timing, and export a JSON / CSV report.
6. Be fully testable with a `FakeModel` and a tiny synthetic dataset so CI stays offline.

The public API should be small:

```python
from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer
from reasonflow.eval import Evaluator, EvalConfig, GSMDataset

eval_cfg = EvalConfig(max_problems=50, metric="numeric_match")
evaluator = Evaluator(engine, eval_cfg)
report = evaluator.run(GSMDataset(split="test"))
report.save_json("results.json")
print(report.accuracy, report.speedup)
```

## 3. Approaches Considered

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| A. Minimal inline evaluator in `examples/eval_demo.py` | Fast to write | Not reusable, no tests, no metrics registry | Rejected |
| B. Pluggable `eval.py` module with `Dataset`, `AnswerExtractor`, `Metric` | Reusable, testable, extensible to new datasets | More code, requires designing stable protocols | **Selected** |
| C. Full tree-of-thought search engine using `RSBCMManager` | Core capability, very hard | Scope too large for a single feature; RSBCM integration is a separate project | Deferred |

Approach B is the sweet spot: it is hard enough to be a substantial contribution, makes the repo much more credible, and does not destabilize the inference engine.

## 4. Architecture

```
reasonflow/eval.py
  EvalConfig          — dataset split, max problems, metric name, output paths
  Evaluator           — orchestrates solve vs baseline over a Dataset
  Dataset (protocol)  — __len__, __getitem__ -> (id, problem, answer)
  HFTextDataset       — wraps a HF datasets.Dataset with column mapping
  AnswerExtractor     — extracts final answer from generated text
  Metric (protocol)   — score(prediction: str, gold: str) -> float
  ExactMatchMetric
  NumericMatchMetric
  ContainsMetric
  EvalResult          — per-problem dataclass
  EvalReport          — aggregate dataclass + save_json / save_csv
```

### 4.1 `Dataset` protocol

```python
class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[str, str, str]: ...
```

`HFTextDataset` maps arbitrary column names (`question`/`answer`, `problem`/`solution`, etc.) to `(id, problem, gold)`. It supports a subset of `max_problems`.

### 4.2 `AnswerExtractor`

- Strips chain-of-thought text after `####` or `Answer:` markers.
- Falls back to last number / last `\boxed{...}` / last sentence.
- Configurable marker list.
- Normalizes whitespace, currency, and punctuation.

### 4.3 `Metric`

- `exact_match`: normalized string equality.
- `numeric_match`: parse both strings as numbers (int/float) and compare within tolerance.
- `contains`: gold substring appears in prediction.

### 4.4 `Evaluator`

```python
def run(self, dataset: Dataset) -> EvalReport:
    for problem_id, problem, gold in dataset:
        rksc_result = self.engine.solve(problem)
        base_result = self.engine.baseline_solve(problem)
        pred = self.extractor.extract(rksc_result.best_text)
        score = self.metric.score(pred, gold)
        results.append(EvalResult(...))
    return EvalReport.from_results(results)
```

- Supports `EngineConfig` re-use for fair comparison.
- Records `total_time_ms` for both modes from `SolveResult`.
- Uses `tqdm` for progress.

### 4.5 `EvalReport`

- `accuracy`: mean score.
- `speedup`: ratio-of-means total baseline / total RKSC time.
- `rksc_ms`, `baseline_ms`: per-problem and total.
- `save_json`, `save_csv`, `__repr__`.

## 5. Testing Strategy

- Unit tests in `tests/test_eval.py`.
- `FakeModel`/`FakeTokenizer` fixtures that produce deterministic text with a known answer marker (e.g. `ANSWER: 42`).
- Tiny in-memory `Dataset` implementation.
- Test each `Metric` and `AnswerExtractor` with edge cases.
- Test `Evaluator.run` end-to-end and assert accuracy == 1.0 and speedup > 0.
- Test `HFTextDataset` column mapping with a mocked HF `datasets.Dataset`.

## 6. Files to Create / Modify

- `src/reasonflow/eval.py` — new module.
- `src/reasonflow/__init__.py` — export public evaluator symbols.
- `tests/test_eval.py` — new tests.
- `examples/eval_demo.py` — demo on a small dataset subset.
- `AGENTS.md` — update test/benchmark baselines after verification.
- `pyproject.toml` — no new dependencies (uses existing `datasets`, `tqdm`).

## 7. Acceptance Criteria

- `ruff check src tests examples` clean.
- `SKIP_ENGINE_TESTS=1 py -3.11 -m pytest -q` passes with `test_eval.py` green.
- `py -3.11 examples/eval_demo.py --max-problems 10` runs offline with a tiny dataset and prints accuracy + speedup.
- `Evaluator` produces identical results on repeated runs with `temperature=0`.

## 8. Out of Scope

- New decoding algorithms (stays in `eval` only).
- Real GSM8K full-run in CI (demo uses tiny subset; full runs are manual).
- Web UI / plotting.
- Distributed evaluation.

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Answer extraction is brittle across model formats | Provide multiple extractors and a configurable marker list; test extensively |
| HF `datasets` offline unavailable in CI | Tests use in-memory `Dataset`; only `examples/eval_demo.py` uses HF datasets |
| `MultiBranchEngine` import may fail in some worktrees | `eval.py` imports `EngineConfig` lazily inside `Evaluator` or uses `try/except` pattern |
| Numeric parsing edge cases (fractions, commas) | Use regex-based normalization and `fractions.Fraction` fallback |

## 10. Atomic Commits (tentative)

1. `feat(eval): add EvalConfig, Dataset protocol, and HFTextDataset`
2. `feat(eval): add AnswerExtractor and Metric registry`
3. `feat(eval): add Evaluator and EvalReport`
4. `test(eval): add FakeModel-based evaluator tests`
5. `examples(eval): add eval_demo.py for GSM8K subset`
6. `chore: update AGENTS.md baselines after eval verification`

---

**Next step:** write implementation plan and execute with TDD.
