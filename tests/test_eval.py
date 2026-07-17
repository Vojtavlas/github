from unittest.mock import MagicMock

import pytest

from reasonflow.eval import (
    AnswerExtractor,
    ContainsMetric,
    EvalConfig,
    EvalReport,
    EvalResult,
    Evaluator,
    ExactMatchMetric,
    HFTextDataset,
    InMemoryDataset,
    NumericMatchMetric,
    get_metric,
)


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


def test_evaluator_runs_offline():
    engine = MagicMock()

    def _make_result(text, ms):
        r = MagicMock()
        r.best_text = text
        r.total_time_ms = ms
        return r

    engine.solve.side_effect = [
        _make_result("#### 4", 100.0),
        _make_result("#### 6", 100.0),
    ]
    engine.baseline_solve.side_effect = [
        _make_result("#### 4", 120.0),
        _make_result("#### 6", 120.0),
    ]

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


def test_public_api_exports():
    from reasonflow import EvalConfig, EvalReport, Evaluator, InMemoryDataset

    assert Evaluator is not None
    assert EvalConfig is not None
    assert EvalReport is not None
    assert InMemoryDataset is not None
