# ReasonFlow Dataset Evaluator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `reasonflow.eval` subsystem that evaluates `MultiBranchEngine` on reasoning datasets, extracts final answers, scores them with pluggable metrics, and exports accuracy/speedup reports.

**Architecture:** A protocol-based `Dataset` wraps HF datasets with column mapping. `AnswerExtractor` and a `Metric` registry handle answer cleaning and scoring. `Evaluator` orchestrates `engine.solve` vs `engine.baseline_solve` over a `Dataset` and produces an `EvalReport` with JSON/CSV export. All components are testable offline with `FakeModel` and in-memory datasets.

**Tech Stack:** Python 3.11, torch>=2.0, transformers>=4.40, datasets>=2.14 (already a dependency), tqdm (already a dependency), pytest, ruff.

## Global Constraints

- Python: `>=3.9`; use `py -3.11` for dev commands.
- `ruff`/`black` line length 100, target `py39`.
- No new heavy dependencies; only existing `datasets`, `tqdm`, `numpy`, `scipy`, `torch`, `transformers`.
- Public API additions go through `src/reasonflow/__init__.py` only after the feature is green.
- Tests must run with `SKIP_ENGINE_TESTS=1` and not require Hugging Face Hub or real model downloads.
- Conventional Commits; do not push/PR unless explicitly asked.

---

## Task 1: `EvalConfig` and `Dataset` protocol

**Files:**
- Create: `src/reasonflow/eval.py` (start with these classes)
- Create: `tests/test_eval.py`

**Interfaces:**
- Consumes: existing `EngineConfig` (for `max_new_tokens` defaults later).
- Produces: `EvalConfig` dataclass, `Dataset` protocol, `InMemoryDataset`, `HFTextDataset`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
import pytest

from reasonflow.eval import EvalConfig, InMemoryDataset


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.max_problems is None
    assert cfg.metric == "exact_match"
    assert cfg.split == "test"


def test_in_memory_dataset():
    data = [
        ("p1", "What is 2+2?", "4"),
        ("p2", "What is 3+3?", "6"),
    ]
    ds = InMemoryDataset(data)
    assert len(ds) == 2
    assert ds[0] == ("p1", "What is 2+2?", "4")


def test_hftext_dataset_maps_columns():
    pytest.skip("HFTextDataset not implemented yet")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_eval.py -v`

Expected: `ModuleNotFoundError` or `ImportError` for `reasonflow.eval`.

- [ ] **Step 3: Write minimal implementation**

Create `src/reasonflow/eval.py` with:

```python
"""Evaluation harness for accuracy and speedup benchmarking."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    max_problems: Optional[int] = None
    metric: str = "exact_match"
    split: str = "test"
    answer_markers: Tuple[str, ...] = ("####", "Answer:", "answer is", "ANSWER:")
    output_json: Optional[str] = None
    output_csv: Optional[str] = None
    seed: int = 42


class Dataset(Protocol):
    """A simple indexable dataset yielding (problem_id, problem, gold_answer)."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        ...


class InMemoryDataset:
    """Wrap a list of (id, problem, gold) tuples."""

    def __init__(self, data: List[Tuple[str, str, str]]) -> None:
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self._data[idx]
```

Leave `HFTextDataset` for Task 2.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_eval.py::test_eval_config_defaults tests/test_eval.py::test_in_memory_dataset -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasonflow/eval.py tests/test_eval.py
git commit -m "feat(eval): add EvalConfig and Dataset protocol"
```

---

## Task 2: `HFTextDataset` and column mapping

**Files:**
- Modify: `src/reasonflow/eval.py`
- Modify: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Dataset` protocol, `EvalConfig`.
- Produces: `HFTextDataset`.

- [ ] **Step 1: Write the failing test**

Replace `test_hftext_dataset_maps_columns` in `tests/test_eval.py` with:

```python
def test_hftext_dataset_maps_columns():
    class FakeHFDataset:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self._rows[idx]
            return self._rows[idx]

        def select(self, indices):
            return FakeHFDataset([self._rows[i] for i in indices])

    fake = FakeHFDataset(
        [
            {"id": "1", "question": "2+2?", "answer": "4"},
            {"id": "2", "problem": "3+3?", "solution": "6"},
        ]
    )
    ds1 = HFTextDataset(fake, id_column="id", problem_column="question", answer_column="answer")
    assert len(ds1) == 2
    assert ds1[0] == ("1", "2+2?", "4")

    ds2 = HFTextDataset(fake, problem_column="problem", answer_column="solution")
    assert ds2[1] == ("2", "3+3?", "6")
```

Run the test; expect `HFTextDataset` not defined.

- [ ] **Step 2: Implement `HFTextDataset`**

Add to `src/reasonflow/eval.py`:

```python
class HFTextDataset:
    """Wrap a Hugging Face ``datasets.Dataset`` with column name mapping."""

    def __init__(
        self,
        hf_dataset: Any,
        id_column: Optional[str] = None,
        problem_column: str = "question",
        answer_column: str = "answer",
        max_problems: Optional[int] = None,
    ) -> None:
        self._data = hf_dataset
        if max_problems is not None and max_problems > 0:
            self._data = self._data.select(list(range(min(max_problems, len(self._data)))))
        self.id_column = id_column
        self.problem_column = problem_column
        self.answer_column = answer_column

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        row = self._data[idx]
        problem_id = str(row.get(self.id_column, idx)) if self.id_column else str(idx)
        problem = str(row[self.problem_column])
        answer = str(row[self.answer_column])
        return problem_id, problem, answer

    @classmethod
    def from_name(
        cls,
        name: str,
        config: Optional[str] = None,
        split: str = "test",
        problem_column: str = "question",
        answer_column: str = "answer",
        id_column: Optional[str] = None,
        max_problems: Optional[int] = None,
    ) -> "HFTextDataset":
        """Load a Hugging Face dataset by name."""
        from datasets import load_dataset

        hf_dataset = load_dataset(name, config, split=split)
        return cls(
            hf_dataset,
            id_column=id_column,
            problem_column=problem_column,
            answer_column=answer_column,
            max_problems=max_problems,
        )
```

- [ ] **Step 3: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_eval.py::test_hftext_dataset_maps_columns -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/reasonflow/eval.py tests/test_eval.py
git commit -m "feat(eval): add HFTextDataset with column mapping"
```

---

## Task 3: `AnswerExtractor` and `Metric` registry

**Files:**
- Modify: `src/reasonflow/eval.py`
- Modify: `tests/test_eval.py`

**Interfaces:**
- Consumes: `EvalConfig` for marker list.
- Produces: `AnswerExtractor`, `Metric` protocol, `ExactMatchMetric`, `NumericMatchMetric`, `ContainsMetric`, `get_metric(name)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_eval.py`:

```python
from reasonflow.eval import (
    AnswerExtractor,
    ContainsMetric,
    ExactMatchMetric,
    NumericMatchMetric,
    get_metric,
)


def test_extract_answer_after_marker():
    ext = AnswerExtractor()
    assert ext.extract("The answer is #### 42 .") == "42"


def test_extract_answer_falls_back_to_last_number():
    ext = AnswerExtractor()
    assert ext.extract("There are 35 chickens and 12 rabbits total 47.") == "47"


def test_extract_answer_empty():
    ext = AnswerExtractor()
    assert ext.extract("") == ""


def test_exact_match_metric():
    m = ExactMatchMetric()
    assert m.score("42", "42") == 1.0
    assert m.score("42", " 42 ") == 1.0
    assert m.score("42", "43") == 0.0


def test_numeric_match_metric():
    m = NumericMatchMetric()
    assert m.score("42", "42") == 1.0
    assert m.score("3.14", "3.140") == 1.0
    assert m.score("1,000", "1000") == 1.0
    assert m.score("42", "43") == 0.0


def test_contains_metric():
    m = ContainsMetric()
    assert m.score("The answer is 42.", "42") == 1.0
    assert m.score("42", "43") == 0.0


def test_get_metric():
    assert isinstance(get_metric("exact_match"), ExactMatchMetric)
    assert isinstance(get_metric("numeric_match"), NumericMatchMetric)
    assert isinstance(get_metric("contains"), ContainsMetric)
    with pytest.raises(ValueError):
        get_metric("unknown")
```

Run the tests; expect import/attribute errors.

- [ ] **Step 2: Implement `AnswerExtractor` and metrics**

Add to `src/reasonflow/eval.py` (before `HFTextDataset` or after, but keep imports updated):

```python
import re
from abc import ABC, abstractmethod


class AnswerExtractor:
    """Extract a final answer string from generated text."""

    def __init__(
        self,
        markers: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.markers = markers or ("####", "Answer:", "answer is", "ANSWER:")

    def extract(self, text: str) -> str:
        """Return the cleaned answer string."""
        text = text.strip()
        if not text:
            return ""

        # Try explicit markers first, taking text after the last marker.
        best = ""
        for marker in self.markers:
            if marker in text:
                tail = text.split(marker)[-1]
                best = self._clean(tail)
                if best:
                    break
        if best:
            return best

        # Try \boxed{...}
        boxed = re.search(r"\\\\boxed\{([^}]+)\}", text)
        if boxed:
            return self._clean(boxed.group(1))

        # Last number fallback.
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return self._clean(numbers[-1])

        return self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        # Remove trailing punctuation.
        text = re.sub(r"[.,;:!?]+$", "", text)
        return text.strip()


class Metric(ABC):
    """Score a prediction against a gold answer."""

    @abstractmethod
    def score(self, prediction: str, gold: str) -> float:
        ...


class ExactMatchMetric(Metric):
    """Case-insensitive, whitespace-normalized exact match."""

    def score(self, prediction: str, gold: str) -> float:
        return 1.0 if self._normalize(prediction) == self._normalize(gold) else 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())


class NumericMatchMetric(Metric):
    """Parse both strings as numbers and compare within tolerance."""

    def __init__(self, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> None:
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def score(self, prediction: str, gold: str) -> float:
        pred_num = self._parse(prediction)
        gold_num = self._parse(gold)
        if pred_num is None or gold_num is None:
            return ExactMatchMetric().score(prediction, gold)
        import math

        return 1.0 if math.isclose(pred_num, gold_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol) else 0.0

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        text = text.strip()
        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = re.sub(r"\s+", "", text)
        try:
            if "/" in text:
                from fractions import Fraction

                return float(Fraction(text))
            return float(text)
        except (ValueError, ZeroDivisionError):
            return None


class ContainsMetric(Metric):
    """Check if the normalized prediction contains the normalized gold answer."""

    def score(self, prediction: str, gold: str) -> float:
        return 1.0 if self._normalize(gold) in self._normalize(prediction) else 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())


_METRICS = {
    "exact_match": ExactMatchMetric,
    "numeric_match": NumericMatchMetric,
    "contains": ContainsMetric,
}


def get_metric(name: str) -> Metric:
    if name not in _METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from {list(_METRICS)}.")
    return _METRICS[name]()
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_eval.py -v`

Expected: all `test_extract_*`, `test_*_metric`, and `test_get_metric` PASS.

- [ ] **Step 4: Commit**

```bash
git add src/reasonflow/eval.py tests/test_eval.py
git commit -m "feat(eval): add AnswerExtractor and Metric registry"
```

---

## Task 4: `EvalResult` and `EvalReport`

**Files:**
- Modify: `src/reasonflow/eval.py`
- Modify: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Metric.score` results and timing data.
- Produces: `EvalResult` dataclass, `EvalReport` dataclass with `from_results`, `save_json`, `save_csv`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from reasonflow.eval import EvalResult, EvalReport


def test_eval_report_aggregates():
    results = [
        EvalResult(
            problem_id="1",
            problem="2+2?",
            gold="4",
            rksc_prediction="4",
            baseline_prediction="4",
            rksc_score=1.0,
            baseline_score=1.0,
            rksc_ms=100.0,
            baseline_ms=120.0,
        ),
        EvalResult(
            problem_id="2",
            problem="3+3?",
            gold="6",
            rksc_prediction="5",
            baseline_prediction="6",
            rksc_score=0.0,
            baseline_score=1.0,
            rksc_ms=200.0,
            baseline_ms=240.0,
        ),
    ]
    report = EvalReport.from_results(results)
    assert report.accuracy == 0.5
    assert report.baseline_accuracy == 1.0
    assert report.speedup == 360.0 / 300.0
    assert len(report.results) == 2


def test_eval_report_save_json(tmp_path):
    result = EvalResult(
        problem_id="1",
        problem="2+2?",
        gold="4",
        rksc_prediction="4",
        baseline_prediction="4",
        rksc_score=1.0,
        baseline_score=1.0,
        rksc_ms=100.0,
        baseline_ms=120.0,
    )
    report = EvalReport.from_results([result])
    out = tmp_path / "report.json"
    report.save_json(str(out))
    import json

    data = json.loads(out.read_text())
    assert data["accuracy"] == 1.0
    assert data["speedup"] == 1.2
    assert len(data["results"]) == 1
```

Run; expect `EvalResult` not defined.

- [ ] **Step 2: Implement `EvalResult` and `EvalReport`**

Add to `src/reasonflow/eval.py`:

```python
import csv
import json
from dataclasses import asdict, dataclass


@dataclass
class EvalResult:
    problem_id: str
    problem: str
    gold: str
    rksc_prediction: str
    baseline_prediction: str
    rksc_score: float
    baseline_score: float
    rksc_ms: float
    baseline_ms: float


@dataclass
class EvalReport:
    accuracy: float
    baseline_accuracy: float
    speedup: float
    rksc_ms: float
    baseline_ms: float
    results: List[EvalResult]

    @classmethod
    def from_results(cls, results: List[EvalResult]) -> "EvalReport":
        if not results:
            return cls(
                accuracy=0.0,
                baseline_accuracy=0.0,
                speedup=1.0,
                rksc_ms=0.0,
                baseline_ms=0.0,
                results=[],
            )
        rksc_scores = [r.rksc_score for r in results]
        baseline_scores = [r.baseline_score for r in results]
        total_rksc = sum(r.rksc_ms for r in results)
        total_baseline = sum(r.baseline_ms for r in results)
        speedup = total_baseline / total_rksc if total_rksc > 0 else float("nan")
        return cls(
            accuracy=sum(rksc_scores) / len(rksc_scores),
            baseline_accuracy=sum(baseline_scores) / len(baseline_scores),
            speedup=speedup,
            rksc_ms=total_rksc,
            baseline_ms=total_baseline,
            results=results,
        )

    def save_json(self, path: str) -> None:
        data = {
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "speedup": self.speedup,
            "rksc_ms": self.rksc_ms,
            "baseline_ms": self.baseline_ms,
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["problem_id", "problem", "gold", "rksc_prediction", "baseline_prediction", "rksc_score", "baseline_score", "rksc_ms", "baseline_ms"])
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `py -3.11 -m pytest tests/test_eval.py::test_eval_report_aggregates tests/test_eval.py::test_eval_report_save_json -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/reasonflow/eval.py tests/test_eval.py
git commit -m "feat(eval): add EvalResult and EvalReport"
```

---

## Task 5: `Evaluator` orchestrator

**Files:**
- Modify: `src/reasonflow/eval.py`
- Modify: `tests/test_eval.py`

**Interfaces:**
- Consumes: `EvalConfig`, `Dataset`, `AnswerExtractor`, `Metric`, `EvalReport`.
- Produces: `Evaluator` with `run(dataset)`.

- [ ] **Step 1: Write the failing end-to-end test**

Add to `tests/test_eval.py`:

```python
from unittest.mock import MagicMock

from reasonflow.eval import Evaluator, EvalConfig, InMemoryDataset


def test_evaluator_runs_offline():
    engine = MagicMock()
    rksc_result = MagicMock()
    rksc_result.best_text = "#### 4"
    rksc_result.total_time_ms = 100.0
    baseline_result = MagicMock()
    baseline_result.best_text = "#### 4"
    baseline_result.total_time_ms = 120.0
    engine.solve.return_value = rksc_result
    engine.baseline_solve.return_value = baseline_result

    cfg = EvalConfig(max_problems=2, metric="exact_match")
    evaluator = Evaluator(engine, cfg)
    dataset = InMemoryDataset(
        [("1", "2+2?", "4"), ("2", "3+3?", "6")]
    )
    report = evaluator.run(dataset)

    assert report.accuracy == 1.0
    assert report.baseline_accuracy == 1.0
    assert report.speedup == 240.0 / 200.0
    assert engine.solve.call_count == 2
    assert engine.baseline_solve.call_count == 2
```

Run; expect `Evaluator` not defined.

- [ ] **Step 2: Implement `Evaluator`**

Add to `src/reasonflow/eval.py`:

```python
class Evaluator:
    """Run a model on a dataset and produce an accuracy/speedup report."""

    def __init__(self, engine: Any, config: Optional[EvalConfig] = None) -> None:
        self.engine = engine
        self.config = config or EvalConfig()
        self.extractor = AnswerExtractor(self.config.answer_markers)
        self.metric = get_metric(self.config.metric)

    def run(self, dataset: Dataset) -> EvalReport:
        """Evaluate ``engine`` over ``dataset`` and return an ``EvalReport``."""
        from tqdm import tqdm

        results: List[EvalResult] = []
        max_problems = self.config.max_problems
        size = len(dataset)
        if max_problems is not None and max_problems > 0:
            size = min(size, max_problems)

        iterator = tqdm(range(size), desc="Evaluating", unit="problem")
        for idx in iterator:
            problem_id, problem, gold = dataset[idx]

            rksc_result = self.engine.solve(problem)
            baseline_result = self.engine.baseline_solve(problem)

            rksc_pred = self.extractor.extract(rksc_result.best_text)
            baseline_pred = self.extractor.extract(baseline_result.best_text)

            rksc_score = self.metric.score(rksc_pred, gold)
            baseline_score = self.metric.score(baseline_pred, gold)

            results.append(
                EvalResult(
                    problem_id=str(problem_id),
                    problem=problem,
                    gold=gold,
                    rksc_prediction=rksc_pred,
                    baseline_prediction=baseline_pred,
                    rksc_score=rksc_score,
                    baseline_score=baseline_score,
                    rksc_ms=rksc_result.total_time_ms,
                    baseline_ms=baseline_result.total_time_ms,
                )
            )

        return EvalReport.from_results(results)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_eval.py::test_evaluator_runs_offline -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/reasonflow/eval.py tests/test_eval.py
git commit -m "feat(eval): add Evaluator orchestrator"
```

---

## Task 6: Export public API and `examples/eval_demo.py`

**Files:**
- Modify: `src/reasonflow/__init__.py`
- Create: `examples/eval_demo.py`
- Modify: `tests/test_eval.py` (add import sanity check)

**Interfaces:**
- Consumes: `Evaluator`, `EvalConfig`, `InMemoryDataset`, `HFTextDataset`, `EvalReport`.
- Produces: public exports in `reasonflow` and a runnable demo.

- [ ] **Step 1: Export evaluator symbols**

Modify `src/reasonflow/__init__.py` to add:

```python
from .eval import (
    AnswerExtractor,
    ContainsMetric,
    EvalConfig,
    EvalReport,
    EvalResult,
    Evaluator,
    ExactMatchMetric,
    HFTextDataset,
    InMemoryDataset,
    Metric,
    NumericMatchMetric,
    get_metric,
)

__all__ = [
    # ... existing entries ...
    "AnswerExtractor",
    "ContainsMetric",
    "EvalConfig",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "ExactMatchMetric",
    "HFTextDataset",
    "InMemoryDataset",
    "Metric",
    "NumericMatchMetric",
    "get_metric",
]
```

- [ ] **Step 2: Add public import test**

Add to `tests/test_eval.py`:

```python
def test_public_api_exports():
    from reasonflow import Evaluator, EvalConfig, EvalReport, InMemoryDataset

    assert Evaluator is not None
    assert EvalConfig is not None
```

- [ ] **Step 3: Create `examples/eval_demo.py`**

Create `examples/eval_demo.py`:

```python
"""Run a small accuracy/speedup evaluation on a reasoning dataset."""

import argparse

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer
from reasonflow.eval import EvalConfig as EvalConfigEval
from reasonflow.eval import Evaluator, HFTextDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReasonFlow on a reasoning dataset.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--metric", default="numeric_match")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default="eval_report.json")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    cfg = EngineConfig(
        branching_factor=args.branching_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    engine = MultiBranchEngine(model, tokenizer, cfg)

    eval_cfg = EvalConfigEval(
        max_problems=args.max_problems,
        metric=args.metric,
        split=args.split,
    )
    dataset = HFTextDataset.from_name(
        args.dataset,
        config=args.dataset_config,
        split=args.split,
        problem_column="question",
        answer_column="answer",
        max_problems=args.max_problems,
    )

    evaluator = Evaluator(engine, eval_cfg)
    report = evaluator.run(dataset)
    print(f"Accuracy:     {report.accuracy:.3f}")
    print(f"Baseline acc: {report.baseline_accuracy:.3f}")
    print(f"Speedup:      {report.speedup:.2f}x")
    print(f"RKSC ms:      {report.rksc_ms:.1f}")
    print(f"Baseline ms:  {report.baseline_ms:.1f}")

    if args.output_json:
        report.save_json(args.output_json)
        print(f"Saved JSON report to {args.output_json}")
    if args.output_csv:
        report.save_csv(args.output_csv)
        print(f"Saved CSV report to {args.output_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run import and lint checks**

Run: `py -3.11 -m pytest tests/test_eval.py::test_public_api_exports -v`
Run: `ruff check src tests examples`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasonflow/__init__.py examples/eval_demo.py tests/test_eval.py
git commit -m "feat(eval): export public API and add eval_demo.py"
```

---

## Task 7: Integration verification and AGENTS.md update

**Files:**
- Modify: `AGENTS.md`
- Modify: `tests/test_eval.py` (if gaps found)

**Interfaces:**
- Consumes: full evaluator subsystem.
- Produces: updated baselines and passing verification.

- [ ] **Step 1: Run full fast suite**

Run (PowerShell):
```powershell
$env:SKIP_ENGINE_TESTS = "1"; py -3.11 -m pytest -q
```

Expected: passes; count should increase by the number of `test_eval.py` tests.

- [ ] **Step 2: Run ruff**

Run: `ruff check src tests examples`

Expected: clean.

- [ ] **Step 3: Run the demo on a tiny dataset (optional, with engine integration)**

If you have a local model and want to sanity-check:
```bash
py -3.11 examples/eval_demo.py --max-problems 5 --model Qwen/Qwen3.5-0.8B --device cuda
```

Expected: prints accuracy and speedup; may be slow.

- [ ] **Step 4: Update `AGENTS.md` test baseline**

Edit `AGENTS.md` line:

```
The current baseline is: `ruff check src tests examples` clean; `SKIP_ENGINE_TESTS=1 py -3.11 -m pytest -q` reports `XXX passed, Y skipped`; a full run `py -3.11 -m pytest -q` reports `ZZZ passed, W skipped`.
```

Replace with actual numbers from Step 1.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md
git commit -m "chore: update test baselines after evaluator verification"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every design requirement maps to a task.
- [ ] **Placeholder scan:** No TODO/TBD/"implement later" in code.
- [ ] **Type consistency:** `Dataset.__getitem__` returns `(str, str, str)` everywhere; `Metric.score` returns `float`.
- [ ] **Public API:** `reasonflow/__init__.py` exports only after Task 6.
- [ ] **Offline CI:** `HFTextDataset.from_name` is never used in unit tests; tests use `InMemoryDataset` and a mocked HF dataset.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-17-dataset-evaluator.md`.

Recommended execution mode: **Subagent-Driven Development** — dispatch one subagent per task (or one per pair of tasks) and review after each commit. The tasks are independent enough to run Tasks 1-4 in parallel, but Task 5 depends on Task 4, Task 6 depends on Task 5, and Task 7 is verification.
