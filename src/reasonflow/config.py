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


@dataclass
class RSBCMConfig:
    """Capacity control for tree-search KV cache."""

    max_blocks: int = 2000


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
