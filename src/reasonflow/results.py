"""Result dataclasses for ReasonFlow solvers."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BranchResult:
    """A single generated reasoning branch."""

    branch_id: int
    prompt: str
    text: str
    full_text: str
    generation_confidence: float
    verification_score: float = 0.0
    verified: bool = False
    early_exit_layer: Optional[int] = None


@dataclass
class SolveResult:
    """Aggregated result for a single problem."""

    problem: str
    best_text: str
    branches: List[BranchResult] = field(default_factory=list)
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    total_time_ms: float = 0.0
    skipped_verification: bool = False
