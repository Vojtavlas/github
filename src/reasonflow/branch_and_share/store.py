"""Persistence for branch_and_share experience packets."""

import json
import warnings
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import List, Union

from .results import (
    BranchMetrics,
    ExperiencePacket,
    HypothesisAttempt,
    InspectRecord,
)


def _packet_to_dict(packet: ExperiencePacket) -> dict:
    """Serialize an ExperiencePacket to a plain dictionary."""
    return asdict(packet)


def _dict_to_packet(data: dict) -> ExperiencePacket:
    """Deserialize a dictionary back to an ExperiencePacket."""
    metrics_data = data["metrics"]
    if metrics_data is None:
        metrics = BranchMetrics()
    elif not isinstance(metrics_data, dict):
        raise TypeError("metrics must be a dict")
    else:
        metrics = BranchMetrics(**metrics_data)

    return ExperiencePacket(
        files_and_symbols_inspected=[
            InspectRecord(**item)
            for item in data.get("files_and_symbols_inspected", [])
        ],
        commands_and_tests_run=data.get("commands_and_tests_run", []),
        modified_files_and_diff=data.get("modified_files_and_diff", ""),
        current_passing_tests=data.get("current_passing_tests", []),
        current_failing_tests=data.get("current_failing_tests", []),
        hypotheses_attempted=[
            HypothesisAttempt(**item)
            for item in data.get("hypotheses_attempted", [])
        ],
        evidence_of_failure=data.get("evidence_of_failure", []),
        useful_discoveries=data.get("useful_discoveries", []),
        recommended_next_actions=data.get("recommended_next_actions", []),
        metrics=metrics,  # type: ignore[arg-type]
    )


class ExperienceStore:
    """Append-only JSONL store for ExperiencePacket objects."""

    def __init__(
        self, path: Union[str, Path], max_history: int = 1000
    ) -> None:
        if max_history < 0:
            raise ValueError("max_history must be >= 0")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history

    def append(self, packet: ExperiencePacket) -> None:
        """Append a packet to the store."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_packet_to_dict(packet), default=str) + "\n")

    def load_all(self) -> List[ExperiencePacket]:
        """Load packets in insertion order, capped to ``max_history``."""
        if not self.path.exists():
            return []
        if self.max_history == 0:
            return []

        packets: deque[ExperiencePacket] = deque(maxlen=self.max_history)

        try:
            with self.path.open("r", encoding="utf-8") as f:
                try:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            packets.append(_dict_to_packet(json.loads(line)))
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            warnings.warn(
                                f"Skipping corrupted experience line: {exc}",
                                UserWarning,
                                stacklevel=2,
                            )
                except OSError as exc:
                    warnings.warn(
                        f"Experience store read interrupted: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
        except OSError as exc:
            warnings.warn(
                f"Experience store read failed: {exc}",
                UserWarning,
                stacklevel=2,
            )

        return list(packets)

    def load_recent(self, n: int = 1) -> List[ExperiencePacket]:
        """Load the most recent ``n`` packets, oldest first."""
        packets = self.load_all()
        return packets[-n:] if n > 0 else []
