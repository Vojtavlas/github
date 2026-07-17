"""Incremental JSONL event stream reader for branch_and_share."""

import json
from typing import Iterable, Iterator

from .protocol import Event, MalformedEventError, _validate_event


class EventStreamReader:
    """Yield validated ``Event`` objects from a stream of text chunks.

    The source is an iterable of arbitrary text chunks. Partial lines are
    buffered until a newline is seen, so a single JSON object may be split
    across any number of chunks.
    """

    def __init__(self, source: Iterable[str], line_offset: int = 0) -> None:
        self.source = source
        self.line_offset = line_offset
        self._buffer = ""

    def __iter__(self) -> Iterator[Event]:
        line_no = self.line_offset
        for chunk in self.source:
            self._buffer += chunk
            while "\n" in self._buffer:
                line, sep, self._buffer = self._buffer.partition("\n")
                line_no += 1
                line = line.rstrip("\r")
                if line.strip():
                    yield self._parse_line(line, line_no)

        if self._buffer:
            line_no += 1
            line = self._buffer.rstrip("\r")
            if line.strip():
                yield self._parse_line(line, line_no)
            self._buffer = ""

    def _parse_line(self, line: str, line_no: int) -> Event:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MalformedEventError(
                f"invalid JSON: {exc}", line_no
            ) from exc
        return _validate_event(data, line_no)
