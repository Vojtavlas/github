"""Evaluation harness for accuracy and speedup benchmarking."""

import csv
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Protocol, Tuple


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

        best = ""
        for marker in self.markers:
            if marker in text:
                tail = text.split(marker)[-1]
                best = self._clean(tail)
                if best:
                    break
        if best:
            return best

        boxed = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return self._clean(boxed.group(1))

        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return self._clean(numbers[-1])

        return self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
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

        return (
            1.0
            if math.isclose(pred_num, gold_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            else 0.0
        )

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
        fieldnames = [
            "problem_id",
            "problem",
            "gold",
            "rksc_prediction",
            "baseline_prediction",
            "rksc_score",
            "baseline_score",
            "rksc_ms",
            "baseline_ms",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))


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
        if self.id_column:
            problem_id = str(row.get(self.id_column, idx))
        else:
            problem_id = str(row.get("id", idx))
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
