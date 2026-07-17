"""Adversarial tests for reasonflow.branch_and_share.ExperienceStore."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from reasonflow.branch_and_share import (
    BranchAndShareConfig,
    BranchMetrics,
    ExperiencePacket,
    ExperienceStore,
)


def _valid_packet_dict(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "files_and_symbols_inspected": [],
        "commands_and_tests_run": ["cmd"],
        "modified_files_and_diff": "",
        "current_passing_tests": [],
        "current_failing_tests": [],
        "hypotheses_attempted": [],
        "evidence_of_failure": [],
        "useful_discoveries": [],
        "recommended_next_actions": [],
        "metrics": {},
    }
    base.update(overrides)
    return base


def _make_packet(commands_and_tests_run=None, **overrides: Any) -> ExperiencePacket:
    kwargs: Dict[str, Any] = {
        "files_and_symbols_inspected": [],
        "commands_and_tests_run": commands_and_tests_run or [],
        "modified_files_and_diff": "",
        "current_passing_tests": [],
        "current_failing_tests": [],
        "hypotheses_attempted": [],
        "evidence_of_failure": [],
        "useful_discoveries": [],
        "recommended_next_actions": [],
        "metrics": BranchMetrics(),
    }
    kwargs.update(overrides)
    return ExperiencePacket(**kwargs)


def _write_jsonl(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, dict):
                f.write(json.dumps(line) + "\n")
            else:
                f.write(line + "\n")


def test_truncated_last_line_skipped(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(
        path,
        [
            _valid_packet_dict(commands_and_tests_run=["good"]),
            '{"metrics": {}, "commands_and_tests_run": ["bad"',
        ],
    )
    store = ExperienceStore(path)
    with pytest.warns(UserWarning, match="corrupted"):
        loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].commands_and_tests_run == ["good"]


def test_first_line_corrupted_remaining_loaded(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(
        path,
        [
            '{"metrics": {}, "commands_and_tests_run": ["bad"',
            _valid_packet_dict(commands_and_tests_run=["good"]),
        ],
    )
    store = ExperienceStore(path)
    with pytest.warns(UserWarning, match="corrupted"):
        loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].commands_and_tests_run == ["good"]


def test_empty_file_loads_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    store = ExperienceStore(path)
    assert store.load_all() == []


def test_blank_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(
        path,
        ["", _valid_packet_dict(commands_and_tests_run=["only"]), "   ", "\n"],
    )
    store = ExperienceStore(path)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].commands_and_tests_run == ["only"]


def test_duplicate_packets_loaded(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "store.jsonl")
    packet = _make_packet(commands_and_tests_run=["cmd"])
    store.append(packet)
    store.append(packet)
    all_packets = store.load_all()
    assert len(all_packets) == 2
    assert all_packets[0].commands_and_tests_run == ["cmd"]
    assert all_packets[1].commands_and_tests_run == ["cmd"]
    assert len(store.load_recent(1)) == 1


def test_packet_missing_metrics_skipped(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    good = _valid_packet_dict(commands_and_tests_run=["good"])
    bad = _valid_packet_dict(commands_and_tests_run=["bad"])
    del bad["metrics"]
    _write_jsonl(path, [good, bad])
    store = ExperienceStore(path)
    with pytest.warns(UserWarning, match="corrupted"):
        loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].commands_and_tests_run == ["good"]


def test_null_metrics_defaults_to_branch_metrics(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(path, [{"metrics": None, "commands_and_tests_run": ["cmd"]}])
    store = ExperienceStore(path)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].metrics == BranchMetrics()
    assert loaded[0].commands_and_tests_run == ["cmd"]


def test_ten_thousand_lines_respects_max_history(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    lines = [
        _valid_packet_dict(commands_and_tests_run=[f"cmd-{i:05d}"])
        for i in range(10_000)
    ]
    _write_jsonl(path, lines)
    store = ExperienceStore(path, max_history=10)
    loaded = store.load_all()
    assert len(loaded) == 10
    assert loaded[0].commands_and_tests_run == ["cmd-09990"]
    assert loaded[-1].commands_and_tests_run == ["cmd-09999"]


def test_max_history_zero_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(path, [_valid_packet_dict()])
    store = ExperienceStore(path, max_history=0)
    assert store.load_all() == []


def test_max_history_caps_loaded_packets(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "store.jsonl", max_history=3)
    for i in range(5):
        store.append(_make_packet(commands_and_tests_run=[f"cmd-{i}"]))
    loaded = store.load_all()
    assert len(loaded) == 3
    assert loaded[0].commands_and_tests_run == ["cmd-2"]
    assert loaded[-1].commands_and_tests_run == ["cmd-4"]


def test_file_deleted_mid_load_returns_partial(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    _write_jsonl(
        path,
        [
            _valid_packet_dict(commands_and_tests_run=["first"]),
            _valid_packet_dict(commands_and_tests_run=["second"]),
        ],
    )
    store = ExperienceStore(path)

    class FlakyFile:
        def __init__(self, lines, fail_after):
            self.lines = lines
            self.index = 0
            self.fail_after = fail_after

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == self.fail_after:
                raise OSError("file vanished")
            if self.index >= len(self.lines):
                raise StopIteration
            line = self.lines[self.index]
            self.index += 1
            return line

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            pass

    class FakePath:
        def __init__(self, real_path, fail_after):
            self._real_path = real_path
            self.parent = real_path.parent
            self.fail_after = fail_after

        def exists(self):
            return self._real_path.exists()

        def open(self, *args, **kwargs):
            with open(self._real_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines(keepends=True)
            return FlakyFile(lines, self.fail_after)

    store.path = FakePath(path, fail_after=1)  # type: ignore[assignment]
    with pytest.warns(UserWarning, match="interrupted"):
        loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].commands_and_tests_run == ["first"]


def test_config_rejects_negative_max_history() -> None:
    with pytest.raises(ValueError, match="max_history"):
        BranchAndShareConfig(max_history=-1)


def test_store_rejects_negative_max_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_history"):
        ExperienceStore(tmp_path / "x.jsonl", max_history=-1)
