"""End-to-end multi-branch reasoning engine."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .asks import ASKSManager
from .cache import RSBCMManager
from .cgee import CGEEAnalyzer
from .config import EngineConfig
from .utils import clone_kv_cache, get_transformer_layers

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


@dataclass
class BranchResult:
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
    problem: str
    best_text: str
    branches: List[BranchResult] = field(default_factory=list)
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    total_time_ms: float = 0.0
    skipped_verification: bool = False
    baseline_time_ms: float = 0.0
    speedup: float = 1.0


class MultiBranchEngine:
    """Generate and score several reasoning branches, sharing prefix KV and
    using CGEE to avoid unnecessary verification work.
    """

    def __init__(self, model, tokenizer, config: Optional[EngineConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self.device = next(model.parameters()).device

        n_layers = len(get_transformer_layers(model))
        unembed = model.get_output_embeddings().weight.data
        self.asks = ASKSManager(self.config.rksc, {})
        self.cgee = CGEEAnalyzer(self.config.rksc, unembed, n_layers)
        self.rsbcm = RSBCMManager(self.config.rsbcm)
        self.hints = DEFAULT_BRANCH_HINTS

    def _shared_prefix(self, problem: str) -> str:
        return (
            "You are a helpful assistant solving a math problem. "
            "Show your reasoning and finish with the final answer.\n\n"
            f"Problem: {problem}\n\nReasoning:"
        )

    def _branch_hint(self, branch_id: int) -> str:
        return self.hints[branch_id % len(self.hints)]

    def _tokenize(self, text: str, add_special_tokens: bool = True):
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.device)

    def _sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the next token and return its probability."""
        if self.config.temperature > 0:
            probs = torch.softmax(logits / self.config.temperature, dim=-1)
            if self.config.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = sorted_probs.cumsum(dim=-1)
                remove = cumsum > self.config.top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                filtered = probs.scatter(-1, sorted_indices, torch.where(remove, 0.0, sorted_probs))
                filtered_sum = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                probs = filtered / filtered_sum
            next_token = torch.multinomial(probs, num_samples=1)
            confidence = probs.gather(-1, next_token).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            confidence = probs.gather(-1, next_token).squeeze(-1)
        return next_token.squeeze(-1), confidence

    def _decode(
        self,
        first_input_ids: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, float, object]:
        """Autoregressively decode up to ``max_new_tokens`` tokens.

        ``first_input_ids`` may be a suffix (when ``past_key_values`` is the
        prefix cache) or the full prompt (when ``past_key_values`` is None).
        """
        generated: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        finished = False
        curr_ids = first_input_ids
        curr_mask = attention_mask
        pkv = past_key_values
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            if finished:
                break
            with torch.inference_mode():
                out = self.model(
                    input_ids=curr_ids,
                    attention_mask=curr_mask,
                    past_key_values=pkv,
                    use_cache=True,
                )
            logits = out.logits[:, -1, :]
            next_token, conf = self._sample(logits)
            generated.append(next_token)
            confidences.append(conf)
            if next_token.item() == eos_id:
                finished = True
                break
            curr_ids = next_token.unsqueeze(-1)
            curr_mask = torch.cat(
                [curr_mask, torch.ones((1, 1), dtype=curr_mask.dtype, device=curr_mask.device)],
                dim=1,
            )
            pkv = out.past_key_values

        if generated:
            sequence = torch.cat([first_input_ids, torch.stack(generated, dim=1)], dim=1)
            mean_conf = torch.stack(confidences).mean().item()
        else:
            sequence = first_input_ids
            mean_conf = 0.0
        return sequence, mean_conf, pkv

    def _continue_generate(
        self,
        first_logits: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, float, object]:
        """Continue autoregressive generation from a pre-computed first-token logit."""
        generated: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        curr_mask = attention_mask
        pkv = past_key_values
        eos_id = self.tokenizer.eos_token_id

        next_token, conf = self._sample(first_logits)
        generated.append(next_token)
        confidences.append(conf)
        if next_token.item() == eos_id or max_new_tokens <= 1:
            generated_ids = torch.stack(generated, dim=1)
            mean_conf = torch.stack(confidences).mean().item()
            return generated_ids, mean_conf, pkv

        curr_ids = next_token.unsqueeze(-1)
        curr_mask = torch.cat(
            [curr_mask, torch.ones((1, 1), dtype=curr_mask.dtype, device=curr_mask.device)],
            dim=1,
        )

        for _ in range(max_new_tokens - 1):
            with torch.inference_mode():
                out = self.model(
                    input_ids=curr_ids,
                    attention_mask=curr_mask,
                    past_key_values=pkv,
                    use_cache=True,
                )
            logits = out.logits[:, -1, :]
            next_token, conf = self._sample(logits)
            generated.append(next_token)
            confidences.append(conf)
            if next_token.item() == eos_id:
                break
            curr_ids = next_token.unsqueeze(-1)
            curr_mask = torch.cat(
                [curr_mask, torch.ones((1, 1), dtype=curr_mask.dtype, device=curr_mask.device)],
                dim=1,
            )
            pkv = out.past_key_values

        generated_ids = torch.stack(generated, dim=1)
        mean_conf = torch.stack(confidences).mean().item()
        return generated_ids, mean_conf, pkv

    def _generate_branch(
        self,
        problem: str,
        branch_id: int,
        prefix_ids: torch.Tensor,
        prefix_pkv,
        prefix_len: int,
    ) -> Tuple[BranchResult, float]:
        """Generate one branch, gating prefix KV reuse with ASKS."""
        prefix_str = self._shared_prefix(problem)
        hint = self._branch_hint(branch_id)
        suffix_ids = self._tokenize("\n" + hint, add_special_tokens=False)["input_ids"]

        prefix_mask = torch.ones((1, prefix_len), dtype=torch.long, device=self.device)
        suffix_mask = torch.ones_like(suffix_ids)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        t0 = time.perf_counter()
        branch_pkv = clone_kv_cache(prefix_pkv)
        with torch.inference_mode():
            prefill_out = self.model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                past_key_values=branch_pkv,
                use_cache=True,
                output_hidden_states=True,
            )
        branch_hidden = prefill_out.hidden_states
        reuse = self.asks.score_branch(branch_id, branch_hidden)

        if not reuse:
            return self._generate_baseline_branch(problem, branch_id)

        if self.config.max_new_tokens == 0:
            generated_ids = suffix_ids.new_empty((1, 0))
            conf = 0.0
        else:
            first_logits = prefill_out.logits[:, -1, :]
            generated_ids, conf, _ = self._continue_generate(
                first_logits,
                prefill_out.past_key_values,
                attention_mask,
                self.config.max_new_tokens,
            )
        gen_ms = (time.perf_counter() - t0) * 1000

        full_sequence = torch.cat([prefix_ids, suffix_ids, generated_ids], dim=1)
        full_text = self.tokenizer.decode(full_sequence[0], skip_special_tokens=True)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt = prefix_str + "\n" + hint

        return (
            BranchResult(
                branch_id=branch_id,
                prompt=prompt,
                text=text,
                full_text=full_text,
                generation_confidence=conf,
            ),
            gen_ms,
        )

    def _generate_baseline_branch(self, problem: str, branch_id: int) -> Tuple[BranchResult, float]:
        """Generate one branch from scratch (no prefix KV sharing)."""
        prefix_str = self._shared_prefix(problem)
        hint = self._branch_hint(branch_id)
        prompt = prefix_str + "\n" + hint
        inputs = self._tokenize(prompt)

        t0 = time.perf_counter()
        sequence, conf, _ = self._decode(
            inputs["input_ids"],
            None,
            inputs["attention_mask"],
            self.config.max_new_tokens,
        )
        gen_ms = (time.perf_counter() - t0) * 1000

        full_text = self.tokenizer.decode(sequence[0], skip_special_tokens=True)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = sequence[:, prompt_len:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return (
            BranchResult(
                branch_id=branch_id,
                prompt=prompt,
                text=text,
                full_text=full_text,
                generation_confidence=conf,
            ),
            gen_ms,
        )

    def _verifier_prompt(self, problem: str, answer: str) -> str:
        return (
            f"Problem: {problem}\n\n"
            f"Proposed answer: {answer}\n\n"
            "Is this answer correct? Answer YES or NO."
        )

    def _verifier_score(self, logits: torch.Tensor) -> float:
        """Score a verifier prompt, returning a probability-like value."""
        yes_ids = self.tokenizer.encode("YES", add_special_tokens=False)
        no_ids = self.tokenizer.encode("NO", add_special_tokens=False)
        yes_id = yes_ids[0] if yes_ids else None
        no_id = no_ids[0] if no_ids else None
        probs = torch.softmax(logits, dim=-1)
        if yes_id is not None and no_id is not None:
            yes_prob = probs[yes_id].item()
            no_prob = probs[no_id].item()
            return yes_prob / (yes_prob + no_prob + 1e-10)
        if yes_id is not None:
            return probs[yes_id].item()
        return 0.5

    def _verify_branch(
        self, problem: str, branch: BranchResult
    ) -> Tuple[float, Optional[int], float]:
        prompt = self._verifier_prompt(problem, branch.text)
        inputs = self._tokenize(prompt)
        t0 = time.perf_counter()
        logits, exit_layer, _ = self.cgee.analyze(
            self.model, inputs["input_ids"], inputs["attention_mask"]
        )
        verify_ms = (time.perf_counter() - t0) * 1000
        score = self._verifier_score(logits[0])
        return score, exit_layer, verify_ms

    def solve(self, problem: str) -> SolveResult:
        """RKSC-style solve: prefix sharing + CGEE-gated verification."""
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
            branch, ms = self._generate_branch(
                problem, b, prefix_inputs["input_ids"], prefix_pkv, prefix_len
            )
            branches.append(branch)
            gen_ms += ms

        # CGEE Level 1: skip verification if one branch is decisively confident.
        confs = [b.generation_confidence for b in branches]
        skip_verification = self.cgee.should_skip_verification(confs)

        verification_ms = 0.0
        if not skip_verification:
            for branch in branches:
                score, exit_layer, ms = self._verify_branch(problem, branch)
                branch.verification_score = score
                branch.verified = True
                branch.early_exit_layer = exit_layer
                verification_ms += ms
            best = max(branches, key=lambda b: b.verification_score)
        else:
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
        t0_total = time.perf_counter()

        branches: List[BranchResult] = []
        gen_ms = 0.0
        for b in range(self.config.branching_factor):
            branch, ms = self._generate_baseline_branch(problem, b)
            branches.append(branch)
            gen_ms += ms

        verification_ms = 0.0
        for branch in branches:
            prompt = self._verifier_prompt(problem, branch.text)
            inputs = self._tokenize(prompt)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.model(**inputs)
            logits = out.logits[:, -1, :]
            verify_ms = (time.perf_counter() - t0) * 1000
            score = self._verifier_score(logits[0])
            branch.verification_score = score
            branch.verified = True
            verification_ms += verify_ms

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
