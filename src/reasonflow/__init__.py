from .asks import ASKSManager
from .cache import RSBCMManager
from .cgee import CGEEAnalyzer
from .config import EngineConfig, RKSCConfig, RSBCMConfig
from .metrics import mean_speedup, speedup
from .results import BranchResult, SolveResult
from .utils import load_model_and_tokenizer

__version__ = "0.1.0"

__all__ = [
    "EngineConfig",
    "RKSCConfig",
    "RSBCMConfig",
    "BranchResult",
    "SolveResult",
    "ASKSManager",
    "CGEEAnalyzer",
    "RSBCMManager",
    "load_model_and_tokenizer",
    "speedup",
    "mean_speedup",
]

# The engine depends on cross-worktree adapter modules (cache_adapter,
# model_adapter) that are not present in this worktree. Import it only when
# available so the rest of the package remains importable and testable.
try:
    from .engine import MultiBranchEngine
    __all__.append(MultiBranchEngine.__name__)
except ImportError:
    pass
