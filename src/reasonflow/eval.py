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
