"""Evaluation harness for accuracy and speedup benchmarking."""

import re
from abc import ABC, abstractmethod
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
