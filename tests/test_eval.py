import pytest

from reasonflow.eval import EvalConfig, HFTextDataset, InMemoryDataset


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
