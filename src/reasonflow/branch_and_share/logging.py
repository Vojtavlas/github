"""Session logging for the branch-and-share layer."""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class _SessionPath:
    """Resolved session directory and file path."""

    repo_root: Path
    file_path: Path


class BranchSessionLogger:
    """Write a JSON-lines session log under ``<repo_root>/.reasonflow/sessions/``.

    The logger is safe to share across branches: writes are serialized with a
    lock and the output path is validated to stay inside ``repo_root``. If no
    ``repo_root`` is provided or the directory cannot be used, the logger is a
    no-op so that fast tests and in-memory runs are unaffected.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> None:
        self._repo_root = repo_root
        self._start = start_time or time.time()
        self._path: Optional[Path] = None
        self._fh: Optional[Any] = None
        self._lock = threading.Lock()

        if self._repo_root is not None:
            resolved = self._resolve_path(self._repo_root)
            if resolved is not None:
                self._path = resolved.file_path
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._fh = self._path.open("w", encoding="utf-8")
                except OSError:
                    # If the filesystem refuses the log, degrade to no-op.
                    self._path = None
                    self._fh = None

    def _resolve_path(self, repo_root: str) -> Optional[_SessionPath]:
        """Return the session file path if it stays inside ``repo_root``."""
        try:
            root = Path(repo_root).expanduser().resolve()
        except (OSError, ValueError):
            return None
        if not root.is_dir():
            return None

        iso = datetime.fromtimestamp(self._start, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S-%f"
        )
        sessions_dir = root / ".reasonflow" / "sessions"
        file_path = sessions_dir / f"{iso}.jsonl"

        try:
            file_path.relative_to(root)
        except ValueError:
            return None
        return _SessionPath(repo_root=root, file_path=file_path)

    def log_event(
        self,
        branch_id: int,
        kind: str,
        elapsed_ms: float,
        stagnation_signals: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one JSON line to the session log."""
        if self._fh is None:
            return
        line = {
            "branch_id": branch_id,
            "kind": kind,
            "timestamp": time.time(),
            "elapsed_ms": elapsed_ms,
            "stagnation_signals": stagnation_signals or [],
            "outcome": outcome,
            "payload": payload or {},
        }
        with self._lock:
            self._fh.write(json.dumps(line, default=str, ensure_ascii=False))
            self._fh.write("\n")
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        if self._fh is not None:
            with self._lock:
                self._fh.close()
            self._fh = None

    @property
    def path(self) -> Optional[Path]:
        """Path to the session log, or ``None`` if logging is disabled."""
        return self._path

    def __enter__(self) -> "BranchSessionLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
