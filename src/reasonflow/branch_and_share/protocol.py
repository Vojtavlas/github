"""JSON event protocol for branch_and_share Pi adapters."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class MalformedEventError(ValueError):
    """Raised when a JSON event is malformed or missing required fields."""

    def __init__(self, message: str, line_no: int = 0) -> None:
        loc = f"Line {line_no}: " if line_no else ""
        super().__init__(f"{loc}{message}")
        self.line_no = line_no


@dataclass
class Event:
    """Base class for all JSON events."""

    kind: str
    line_no: int = 0


@dataclass
class ToolCallEvent(Event):
    """A tool invocation event."""

    name: str = ""
    args: Optional[Dict[str, Any]] = None
    result: Any = None
    failed: bool = False
    tokens: int = 0


@dataclass
class FileReadEvent(Event):
    """A file read event."""

    path: str = ""
    symbols: Optional[List[str]] = None


@dataclass
class CommandEvent(Event):
    """A shell command event."""

    command: str = ""
    output: str = ""
    exit_code: int = 0


@dataclass
class TestEvent(Event):
    """A test execution event."""

    name: str = ""
    passed: bool = True
    output: str = ""


@dataclass
class FileChangeEvent(Event):
    """A file change event."""

    path: str = ""
    change_type: str = "modified"
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None


@dataclass
class ModelCallEvent(Event):
    """A model call event."""

    tokens: int = 0


@dataclass
class TokenUsageEvent(Event):
    """A raw token usage event."""

    n: int = 0


@dataclass
class HypothesisEvent(Event):
    """A hypothesis attempt event."""

    description: str = ""
    action: str = ""
    expected_test_change: str = ""
    observed: str = ""
    failed: bool = True


@dataclass
class DiscoveryEvent(Event):
    """A useful discovery event."""

    text: str = ""


@dataclass
class StatusEvent(Event):
    """A terminal status event."""

    status: str = ""
    result: Any = None
    error: str = ""


def _require_str(data: Dict[str, Any], key: str, line_no: int) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise MalformedEventError(f"'{key}' must be a string", line_no)
    return value


def _require_int(data: Dict[str, Any], key: str, line_no: int, default: int = 0) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise MalformedEventError(f"'{key}' must be an integer", line_no)
    return value


def _require_bool(data: Dict[str, Any], key: str, line_no: int, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise MalformedEventError(f"'{key}' must be a boolean", line_no)
    return value


def _optional_str(data: Dict[str, Any], key: str, line_no: int = 0) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MalformedEventError(f"'{key}' must be a string or null", line_no)
    return value


def _optional_str_list(
    data: Dict[str, Any], key: str, line_no: int = 0
) -> Optional[List[str]]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise MalformedEventError(f"'{key}' must be a list of strings", line_no)
    return value


def _validate_event(data: Dict[str, Any], line_no: int = 0) -> Event:
    """Validate a JSON dict and return a concrete ``Event`` instance.

    Raises ``MalformedEventError`` with the line number when the dict is
    missing ``kind`` or required fields for the kind.
    """
    if not isinstance(data, dict):
        raise MalformedEventError("event is not a JSON object", line_no)

    kind = data.get("kind")
    if not isinstance(kind, str) or not kind:
        raise MalformedEventError("missing or invalid 'kind'", line_no)

    common = {"kind": kind, "line_no": line_no}

    if kind == "tool_call":
        args = data.get("args")
        if args is not None and not isinstance(args, dict):
            raise MalformedEventError(
                f"'args' must be a JSON object or null, got {type(args).__name__}",
                line_no,
            )
        return ToolCallEvent(
            name=_require_str(data, "name", line_no),
            args=args,
            result=data.get("result"),
            failed=_require_bool(data, "failed", line_no, False),
            tokens=_require_int(data, "tokens", line_no, 0),
            **common,
        )
    if kind == "file_read":
        return FileReadEvent(
            path=_require_str(data, "path", line_no),
            symbols=_optional_str_list(data, "symbols", line_no),
            **common,
        )
    if kind == "command":
        return CommandEvent(
            command=_require_str(data, "command", line_no),
            output=_optional_str(data, "output", line_no) or "",
            exit_code=_require_int(data, "exit_code", line_no, 0),
            **common,
        )
    if kind == "test":
        return TestEvent(
            name=_require_str(data, "name", line_no),
            passed=_require_bool(data, "passed", line_no, True),
            output=_optional_str(data, "output", line_no) or "",
            **common,
        )
    if kind == "file_change":
        return FileChangeEvent(
            path=_require_str(data, "path", line_no),
            change_type=_optional_str(data, "change_type", line_no) or "modified",
            old_hash=_optional_str(data, "old_hash", line_no),
            new_hash=_optional_str(data, "new_hash", line_no),
            **common,
        )
    if kind == "model_call":
        return ModelCallEvent(
            tokens=_require_int(data, "tokens", line_no, 0),
            **common,
        )
    if kind == "token_usage":
        return TokenUsageEvent(
            n=_require_int(data, "n", line_no, 0),
            **common,
        )
    if kind == "hypothesis":
        return HypothesisEvent(
            description=_require_str(data, "description", line_no),
            action=_require_str(data, "action", line_no),
            expected_test_change=_require_str(data, "expected", line_no),
            observed=_optional_str(data, "observed", line_no) or "",
            failed=_require_bool(data, "failed", line_no, True),
            **common,
        )
    if kind == "discovery":
        return DiscoveryEvent(
            text=_require_str(data, "text", line_no),
            **common,
        )
    if kind == "status":
        status = _require_str(data, "status", line_no)
        if status not in {"success", "error", "stagnation"}:
            raise MalformedEventError(
                f"invalid 'status' value: {status!r}", line_no
            )
        return StatusEvent(
            status=status,
            result=data.get("result"),
            error=_optional_str(data, "error", line_no) or "",
            **common,
        )

    raise MalformedEventError(f"unknown event kind: {kind!r}", line_no)
