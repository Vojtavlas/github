from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RKSCConfig:
    """Hyper-parameters for RKSC's two core mechanisms."""

    # ASKS: hidden-state cosine similarity gate.
    tau: float = 0.75

    # CGEE: entropy early-exit threshold and stability checks.
    theta: float = 8.0
    min_exit_layer: int = 2
    entropy_stability_eps: float = 3.0

    # CGEE Level 1: generation-confidence verification skip.
    gen_conf_threshold: float = 0.70
    gen_conf_gap: float = 0.06
    use_relative_gap: bool = True
    relative_gap_threshold: float = 0.08

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        if self.theta < 0:
            raise ValueError("theta must be >= 0")
        if self.min_exit_layer < 0:
            raise ValueError("min_exit_layer must be >= 0")
        if self.entropy_stability_eps < 0:
            raise ValueError("entropy_stability_eps must be >= 0")
        if not 0 <= self.gen_conf_threshold <= 1:
            raise ValueError("gen_conf_threshold must be in [0, 1]")
        if self.gen_conf_gap < 0:
            raise ValueError("gen_conf_gap must be >= 0")
        if not 0 <= self.relative_gap_threshold <= 1:
            raise ValueError("relative_gap_threshold must be in [0, 1]")


@dataclass
class RSBCMConfig:
    """Capacity control for tree-search KV cache."""

    max_blocks: int = 2000

    def __post_init__(self) -> None:
        if self.max_blocks < 0:
            raise ValueError("max_blocks must be >= 0")


@dataclass
class EngineConfig:
    """Top-level configuration for a ReasonFlow solver."""

    rksc: RKSCConfig = field(default_factory=RKSCConfig)
    rsbcm: RSBCMConfig = field(default_factory=RSBCMConfig)
    branching_factor: int = 2
    max_new_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.95
    max_seq_len: int = 2048
    device: Optional[str] = None
    use_batched_decoding: bool = True

    def __post_init__(self) -> None:
        if self.branching_factor <= 0:
            raise ValueError("branching_factor must be > 0")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_seq_len < 0:
            raise ValueError("max_seq_len must be >= 0")
        if not isinstance(self.use_batched_decoding, bool):
            raise ValueError("use_batched_decoding must be a bool")
