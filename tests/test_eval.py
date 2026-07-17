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
