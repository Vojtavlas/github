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
