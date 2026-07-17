"""Generate a single reasoning branch, with ASKS gating and baseline fallback."""

from typing import Callable, Optional

import torch

from .config import EngineConfig
from .decoder import Decoder
from .results import BranchResult
from .utils import clone_kv_cache


class BranchGenerator:
    """Generate one branch, gating prefix KV reuse with ASKS."""

    def __init__(
        self,
        model,
        tokenizer,
        asks,
        decoder: Decoder,
        config: EngineConfig,
        clone_kv_cache_fn: Optional[Callable] = None,
        shared_prefix: Optional[Callable[[str], str]] = None,
        branch_hint: Optional[Callable[[int], str]] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.asks = asks
        self.decoder = decoder
        self.config = config
        self.clone_kv_cache = clone_kv_cache_fn or clone_kv_cache
        self.shared_prefix = shared_prefix or self._default_shared_prefix
        self.branch_hint = branch_hint or self._default_branch_hint
        if device is not None:
            self.device = device
        elif config.device is not None:
            self.device = config.device
        else:
            self.device = next(model.parameters()).device

    @staticmethod
    def _default_shared_prefix(problem: str) -> str:
        return (
            "You are a helpful assistant solving a math problem. "
            "Show your reasoning and finish with the final answer.\n\n"
            f"Problem: {problem}\n\nReasoning:"
        )

    @staticmethod
    def _default_branch_hint(branch_id: int) -> str:
        return f"Approach {branch_id + 1}: think step by step."

    def _tokenize(self, text: str, add_special_tokens: bool = True) -> dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.device)

    def _generate_baseline_branch(self, problem: str, branch_id: int) -> BranchResult:
        """Generate one branch from scratch (no prefix KV sharing)."""
        prefix_str = self.shared_prefix(problem)
        hint = self.branch_hint(branch_id)
        prompt = prefix_str + "\n" + hint
        inputs = self._tokenize(prompt)

        sequence, conf, _ = self.decoder.decode(
            inputs["input_ids"],
            None,
            inputs["attention_mask"],
            self.config.max_new_tokens,
        )

        full_text = self.tokenizer.decode(sequence[0], skip_special_tokens=True)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = sequence[:, prompt_len:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return BranchResult(
            branch_id=branch_id,
            prompt=prompt,
            text=text,
            full_text=full_text,
            generation_confidence=conf,
        )

    def generate(
        self,
        problem: str,
        branch_id: int,
        prefix_ids: torch.Tensor,
        prefix_pkv,
        prefix_len: int,
    ) -> BranchResult:
        """Generate one branch, reusing prefix KV when ASKS allows it."""
        prefix_str = self.shared_prefix(problem)
        hint = self.branch_hint(branch_id)
        suffix_ids = self._tokenize("\n" + hint, add_special_tokens=False)["input_ids"]

        prefix_mask = torch.ones((1, prefix_len), dtype=torch.long, device=self.device)
        suffix_mask = torch.ones_like(suffix_ids)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        branch_pkv = self.clone_kv_cache(prefix_pkv)
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
            generated_ids, conf, _ = self.decoder.continue_generate(
                first_logits,
                prefill_out.past_key_values,
                attention_mask,
                self.config.max_new_tokens,
            )

        full_sequence = torch.cat([prefix_ids, suffix_ids, generated_ids], dim=1)
        full_text = self.tokenizer.decode(full_sequence[0], skip_special_tokens=True)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt = prefix_str + "\n" + hint

        return BranchResult(
            branch_id=branch_id,
            prompt=prompt,
            text=text,
            full_text=full_text,
            generation_confidence=conf,
        )

    def generate_baseline_branch(self, problem: str, branch_id: int) -> BranchResult:
        """Public alias for baseline branch generation."""
        return self._generate_baseline_branch(problem, branch_id)
