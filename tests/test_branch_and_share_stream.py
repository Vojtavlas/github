"""Tests for reasonflow.branch_and_share.protocol and stream modules."""

import json
from typing import List

import pytest

from reasonflow.branch_and_share.protocol import (
    MalformedEventError,
    ToolCallEvent,
    _validate_event,
)
from reasonflow.branch_and_share.stream import EventStreamReader


def _chunks(text: str, sizes: List[int]) -> List[str]:
    """Split ``text`` into consecutive chunks of the given sizes."""
    chunks = []
    idx = 0
    for size in sizes:
        if idx >= len(text):
            break
        chunks.append(text[idx : idx + size])
        idx += len(chunks[-1])
    return chunks


def test_event_stream_reader_whole_lines() -> None:
    lines = [
        {"kind": "tool_call", "name": "read_file", "tokens": 12},
        {"kind": "command", "command": "pytest", "output": "ok", "exit_code": 0},
        {"kind": "status", "status": "success", "result": "done"},
    ]
    source = "\n".join(json.dumps(line) for line in lines)
    reader = EventStreamReader([source])
    events = list(reader)

    assert len(events) == 3
    assert isinstance(events[0], ToolCallEvent)
    assert events[0].name == "read_file"
    assert events[0].tokens == 12
    assert events[1].command == "pytest"
    assert events[1].exit_code == 0
    assert events[2].status == "success"
    assert events[2].result == "done"
    assert events[2].line_no == 3


def test_event_stream_reader_split_line() -> None:
    text = '{"kind":"tool_call","name":"x","tokens":1}\n'
    chunks = _chunks(text, [10, 12, 100])
    reader = EventStreamReader(chunks)
    events = list(reader)

    assert len(events) == 1
    assert events[0].kind == "tool_call"
    assert events[0].name == "x"
    assert events[0].line_no == 1


def test_event_stream_reader_missing_trailing_newline() -> None:
    text = '{"kind":"discovery","text":"found bug"}'
    reader = EventStreamReader([text])
    events = list(reader)

    assert len(events) == 1
    assert events[0].kind == "discovery"
    assert events[0].text == "found bug"
    assert events[0].line_no == 1


def test_event_stream_reader_crlf_and_empty_lines() -> None:
    text = '\r\n{"kind":"test","name":"t","passed":false}\r\n\r\n'
    reader = EventStreamReader([text])
    events = list(reader)

    assert len(events) == 1
    assert events[0].kind == "test"
    assert events[0].passed is False
    assert events[0].line_no == 2


def test_event_stream_reader_invalid_json() -> None:
    reader = EventStreamReader(['{"kind": "test"}, broken'])
    with pytest.raises(MalformedEventError) as exc_info:
        list(reader)
    assert "invalid JSON" in str(exc_info.value)
    assert exc_info.value.line_no == 1


def test_event_stream_reader_missing_kind() -> None:
    reader = EventStreamReader(['{"name": "read_file"}'])
    with pytest.raises(MalformedEventError) as exc_info:
        list(reader)
    assert "missing or invalid 'kind'" in str(exc_info.value)


def test_event_stream_reader_missing_required_field() -> None:
    reader = EventStreamReader(['{"kind": "command"}'])
    with pytest.raises(MalformedEventError) as exc_info:
        list(reader)
    assert "'command' must be a string" in str(exc_info.value)


def test_event_stream_reader_unknown_kind() -> None:
    reader = EventStreamReader(['{"kind": "foo"}'])
    with pytest.raises(MalformedEventError) as exc_info:
        list(reader)
    assert "unknown event kind" in str(exc_info.value).lower()


def test_event_stream_reader_line_numbers_increment() -> None:
    text = "a\n{\"kind\": \"tool_call\", \"name\": \"n\", \"tokens\": 0}\nb\n"
    reader = EventStreamReader([text])
    with pytest.raises(MalformedEventError) as exc_info:
        list(reader)
    assert "Line 1" in str(exc_info.value)
    assert "invalid JSON" in str(exc_info.value)


def test_validate_event_missing_kind() -> None:
    with pytest.raises(MalformedEventError) as exc_info:
        _validate_event({"name": "read_file"})
    assert "missing or invalid 'kind'" in str(exc_info.value)


def test_validate_event_missing_required_string() -> None:
    with pytest.raises(MalformedEventError) as exc_info:
        _validate_event({"kind": "discovery"})
    assert "'text' must be a string" in str(exc_info.value)


def test_validate_event_status_with_invalid_value() -> None:
    with pytest.raises(MalformedEventError) as exc_info:
        _validate_event({"kind": "status", "status": "unknown"})
    assert "invalid 'status' value" in str(exc_info.value)
