"""End-to-end multi-branch reasoning engine."""

import time
from typing import Dict, List, Optional, cast

import torch

from .asks import ASKSManager
from .branch_generator import BranchGenerator
from .cache import RSBCMManager
from .cache_adapter import clone_kv_cache
from .cgee import CGEEAnalyzer
from .config import EngineConfig
from .decoder import Decoder
from .model_adapter import get_transformer_layers
from .results import BranchResult, SolveResult
from .sampler import Sampler
from .verifier import Verifier

DEFAULT_BRANCH_HINTS = [
    "Approach 1: think step by step.",
    "Approach 2: list all variables and equations.",
    "Approach 3: use a concrete example.",
    "Approach 4: work backwards from the goal.",
    "Approach 5: check units and limiting cases.",
    "Approach 6: draw a diagram mentally.",
    "Approach 7: apply first principles.",
    "Approach 8: enumerate cases and compare.",
]


def _shared_prefix(problem: str) -> str:
    return (
        "You are a helpful assistant solving a math problem. "
        "Show your reasoning and finish with the final answer.\n\n"
        f"Problem: {problem}\n\nReasoning:"
    )


def _branch_hint(branch_id: int, hints: List[str] = DEFAULT_BRANCH_HINTS) -> str:
    return hints[branch_id % len(hints)]


class MultiBranchEngine:
    """Generate and score several reasoning branches, sharing prefix KV and
    using CGEE to avoid unnecessary verification work.
    """

    def __init__(self, model, tokenizer, config: Optional[EngineConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self.device = self.config.device or next(model.parameters()).device

        n_layers = len(get_transformer_layers(model))
        unembed = model.get_output_embeddings().weight.data
        self.asks = ASKSManager(self.config.rksc, {})
        self.cgee = CGEEAnalyzer(self.config.rksc, unembed, n_layers)
        self.rsbcm = RSBCMManager(self.config.rsbcm)

        self.sampler = Sampler(self.config)
        self.decoder = Decoder(self.model, self.tokenizer, self.sampler, self.config)
        self.branch_generator = BranchGenerator(
            self.model,
            self.tokenizer,
            self.asks,
            self.decoder,
            self.config,
            clone_kv_cache_fn=clone_kv_cache,
            shared_prefix=_shared_prefix,
            branch_hint=lambda bid: _branch_hint(bid, DEFAULT_BRANCH_HINTS),
            device=self.device,
        )
        self.verifier = Verifier(
            self.model, self.tokenizer, self.cgee, self.config, device=self.device
        )
        self.hints = DEFAULT_BRANCH_HINTS

    def _shared_prefix(self, problem: str) -> str:
        return _shared_prefix(problem)

    def _branch_hint(self, branch_id: int) -> str:
        return _branch_hint(branch_id, self.hints)

    def _tokenize(self, text: str, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        return cast(
            Dict[str, torch.Tensor],
            self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=self.config.max_seq_len,
            ).to(self.device),
        )

    def solve(self, problem: str) -> SolveResult:
        """RKSC-style solve: prefix sharing + CGEE-gated verification."""
        if self.config.branching_factor <= 0:
            raise ValueError(
                f"branching_factor must be a positive integer, got {self.config.branching_factor}"
            )
        self.asks.reset()
        t0_total = time.perf_counter()

        # Root prefix forward: compute KV + hidden states once.
        t0 = time.perf_counter()
        prefix_str = self._shared_prefix(problem)
        prefix_inputs = self._tokenize(prefix_str)
        prefix_len = prefix_inputs["input_ids"].shape[1]
        with torch.inference_mode():
            root_out = self.model(
                **prefix_inputs,
                use_cache=True,
                output_hidden_states=True,
            )
        prefix_pkv = root_out.past_key_values
        self.asks.capture_root(prefix_pkv, root_out.hidden_states)
        prefix_ms = (time.perf_counter() - t0) * 1000

        branches: List[BranchResult] = []
        gen_ms = 0.0
        for b in range(self.config.branching_factor):
            t_b = time.perf_counter()
            branch = self.branch_generator.generate(
                problem, b, prefix_inputs["input_ids"], prefix_pkv, prefix_len
            )
            gen_ms += (time.perf_counter() - t_b) * 1000
            branches.append(branch)

        # CGEE Level 1: skip verification if one branch is decisively confident.
        confs = [b.generation_confidence for b in branches]
        skip_verification = self.cgee.should_skip_verification(confs)

        verification_ms = 0.0
        if not skip_verification:
            for branch in branches:
                score, exit_layer, verify_ms = self.verifier.verify(problem, branch.text)
                branch.verification_score = score
                branch.verified = True
                branch.early_exit_layer = exit_layer
                verification_ms += verify_ms
            # Ties are stable: max() returns the first maximal branch, which
            # is the lowest branch_id because branches are generated in order.
            best = max(branches, key=lambda b: b.verification_score)
        else:
            # Ties are stable: max() returns the first maximal branch, which
            # is the lowest branch_id because branches are generated in order.
            best = max(branches, key=lambda b: b.generation_confidence)

        total_ms = (time.perf_counter() - t0_total) * 1000
        return SolveResult(
            problem=problem,
            best_text=best.text,
            branches=branches,
            generation_time_ms=gen_ms + prefix_ms,
            verification_time_ms=verification_ms,
            total_time_ms=total_ms,
            skipped_verification=skip_verification,
        )

    def baseline_solve(self, problem: str) -> SolveResult:
        """Naive baseline: generate each branch independently + full verification."""
        if self.config.branching_factor <= 0:
            raise ValueError(
                f"branching_factor must be a positive integer, got {self.config.branching_factor}"
            )
        self.asks.reset()
        t0_total = time.perf_counter()

        branches: List[BranchResult] = []
        gen_ms = 0.0
        for b in range(self.config.branching_factor):
            t_b = time.perf_counter()
            branch = self.branch_generator.generate_baseline_branch(problem, b)
            gen_ms += (time.perf_counter() - t_b) * 1000
            branches.append(branch)

        verification_ms = 0.0
        for branch in branches:
            score, exit_layer, verify_ms = self.verifier.verify(problem, branch.text)
            branch.verification_score = score
            branch.verified = True
            branch.early_exit_layer = exit_layer
            verification_ms += verify_ms

        # Ties are stable: max() returns the first maximal branch, which
        # is the lowest branch_id because branches are generated in order.
        best = max(branches, key=lambda b: b.verification_score)
        total_ms = (time.perf_counter() - t0_total) * 1000
        return SolveResult(
            problem=problem,
            best_text=best.text,
            branches=branches,
            generation_time_ms=gen_ms,
            verification_time_ms=verification_ms,
            total_time_ms=total_ms,
            skipped_verification=False,
        )
