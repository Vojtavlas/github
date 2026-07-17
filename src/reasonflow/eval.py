"""Evaluation harness for accuracy and speedup benchmarking."""

import csv
import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Protocol, Tuple

import torch


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    max_problems: Optional[int] = None
    metric: str = "exact_match"
    split: str = "test"
    answer_markers: Tuple[str, ...] = ("####", "Answer:", "ANSWER:")
    output_json: Optional[str] = None
    output_csv: Optional[str] = None
    seed: int = 42
    warmup: int = 1
    runs: int = 3
    extract_gold: bool = True


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
    """Extract a final answer string from generated or gold text."""

    def __init__(
        self,
        markers: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.markers = markers or ("####", "Answer:", "ANSWER:")

    def extract(self, text: str) -> str:
        """Return the cleaned answer string."""
        text = text.strip()
        if not text:
            return ""

        lower_text = text.lower()
        last_pos = -1
        last_marker = ""
        for marker in self.markers:
            pos = lower_text.rfind(marker.lower())
            if pos > last_pos:
                last_pos = pos
                last_marker = marker
        if last_pos != -1:
            suffix = text[last_pos + len(last_marker) :]
            cleaned = self._clean(suffix)
            if cleaned:
                return cleaned

        boxed = self._extract_boxed(text)
        if boxed is not None:
            cleaned = self._clean(self._parse_latex_boxed(boxed))
            if cleaned:
                return cleaned

        frac = self._parse_frac(text)
        if frac is not None:
            return frac

        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return self._clean(numbers[-1])

        return self._clean(text)

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        """Extract the contents of the outermost \boxed{...}, handling one nesting level."""
        start = text.find("\\boxed{")
        if start == -1:
            return None
        depth = 0
        content_start = start + len("\\boxed{")
        for i in range(content_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[content_start:i]
                depth -= 1
        return None

    @classmethod
    def _parse_latex_boxed(cls, text: str) -> str:
        """Clean LaTeX inside a box and return a parseable string."""
        # Handle \frac{a}{b} inside the box.
        frac = cls._parse_frac(text)
        if frac is not None:
            return frac
        return text

    @staticmethod
    def _parse_frac(text: str) -> Optional[str]:
        """Convert a simple \\frac{a}{b} to 'a/b' for numeric parsing."""
        match = re.search(r"\\frac\{([^}]+)\}\{([^}]+)\}", text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return None

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        # Remove trailing sentence punctuation.
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
        if not gold.strip() and not prediction.strip():
            return 1.0
        if not gold.strip() or not prediction.strip():
            return 0.0
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
        text = re.sub(r"\s+", "", text)

        has_percent = "%" in text
        text = text.replace("%", "")

        try:
            if "/" in text:
                from fractions import Fraction

                value = float(Fraction(text))
            else:
                value = float(text)
            if has_percent and "/" not in text:
                value /= 100.0
            return value
        except (ValueError, ZeroDivisionError):
            return None


class ContainsMetric(Metric):
    """Check if the normalized prediction contains the normalized gold answer."""

    def score(self, prediction: str, gold: str) -> float:
        if not gold.strip():
            return 0.0 if prediction.strip() else 1.0
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
    error: Optional[str] = None


@dataclass
class EvalReport:
    accuracy: float
    baseline_accuracy: float
    speedup: Optional[float]
    rksc_ms: float
    baseline_ms: float
    results: List[EvalResult]

    @classmethod
    def from_results(cls, results: List[EvalResult]) -> "EvalReport":
        if not results:
            return cls(
                accuracy=0.0,
                baseline_accuracy=0.0,
                speedup=None,
                rksc_ms=0.0,
                baseline_ms=0.0,
                results=[],
            )
        rksc_scores = [r.rksc_score for r in results]
        baseline_scores = [r.baseline_score for r in results]
        total_rksc = sum(r.rksc_ms for r in results)
        total_baseline = sum(r.baseline_ms for r in results)
        speedup = total_baseline / total_rksc if total_rksc > 0 else None
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)

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
            "error",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
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

    def _extract(self, text: str) -> str:
        return self.extractor.extract(text)

    def _call_and_time(self, method, problem: str, seed_value: int) -> Tuple[str, float]:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = method(problem)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        text = getattr(result, "best_text", "")
        return text, (t1 - t0) * 1000.0

    def _run_pair(
        self, problem: str, idx: int
    ) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float], Optional[str]]:
        warmup = max(0, self.config.warmup)
        runs = max(1, self.config.runs)
        solve = self.engine.solve
        baseline_solve = self.engine.baseline_solve
        base_seed = self.config.seed + idx * 1000

        try:
            for w in range(warmup):
                torch.manual_seed(base_seed + w)
                solve(problem)
                torch.manual_seed(base_seed + warmup + w)
                baseline_solve(problem)

            rksc_times: List[float] = []
            baseline_times: List[float] = []
            rksc_text = ""
            baseline_text = ""
            for r in range(runs):
                rksc_seed = base_seed + 2 * warmup + r
                baseline_seed = base_seed + 2 * warmup + runs + r
                if (idx + r) % 2 == 0:
                    rksc_text, ms = self._call_and_time(solve, problem, rksc_seed)
                    rksc_times.append(ms)
                    baseline_text, ms = self._call_and_time(baseline_solve, problem, baseline_seed)
                    baseline_times.append(ms)
                else:
                    baseline_text, ms = self._call_and_time(baseline_solve, problem, baseline_seed)
                    baseline_times.append(ms)
                    rksc_text, ms = self._call_and_time(solve, problem, rksc_seed)
                    rksc_times.append(ms)
        except Exception as exc:
            return None, None, None, None, str(exc)

        rksc_ms = sum(rksc_times) / len(rksc_times) if rksc_times else None
        baseline_ms = sum(baseline_times) / len(baseline_times) if baseline_times else None
        return rksc_text, rksc_ms, baseline_text, baseline_ms, None

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
            problem_id, problem, raw_gold = dataset[idx]
            gold = self._extract(raw_gold) if self.config.extract_gold else raw_gold

            rksc_text, rksc_ms, baseline_text, baseline_ms, error = self._run_pair(problem, idx)

            if error is not None:
                results.append(
                    EvalResult(
                        problem_id=str(problem_id),
                        problem=problem,
                        gold=gold,
                        rksc_prediction="",
                        baseline_prediction="",
                        rksc_score=0.0,
                        baseline_score=0.0,
                        rksc_ms=0.0,
                        baseline_ms=0.0,
                        error=error,
                    )
                )
                continue

            rksc_pred = self._extract(rksc_text) if rksc_text is not None else ""
            baseline_pred = self._extract(baseline_text) if baseline_text is not None else ""

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
                    rksc_ms=rksc_ms if rksc_ms is not None else 0.0,
                    baseline_ms=baseline_ms if baseline_ms is not None else 0.0,
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
