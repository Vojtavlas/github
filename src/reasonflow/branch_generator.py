"""Generate a single reasoning branch, with ASKS gating and baseline fallback."""

from typing import Any, Callable, Dict, List, Optional, cast

import torch

from .cache_adapter import clone_kv_cache, expand_kv, get_cache_adapter, select_kv_cache_rows
from .config import EngineConfig
from .decoder import Decoder
from .results import BranchResult


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

        self._patch_linear_attn_chunk_size(model)

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

    @staticmethod
    def _patch_linear_attn_chunk_size(model) -> None:
        """Shrink the chunk size used by the torch fallback for gated delta nets.

        The fallback ``torch_chunk_gated_delta_rule`` defaults to a chunk size of
        64, which pads short branch suffixes up to 64 positions.  Power-of-two
        chunk sizes up to 64 are mathematically equivalent for the fallback, so
        for short sequences we can use the next power of two and avoid most of
        the padding overhead.
        """
        for module in model.modules():
            fn = getattr(module, "chunk_gated_delta_rule", None)
            if fn is None or getattr(fn, "_reasonflow_patched", False):
                continue
            if getattr(fn, "__name__", None) != "torch_chunk_gated_delta_rule":
                continue

            def wrapper(*args, _orig=fn, **kwargs):
                if "chunk_size" not in kwargs and len(args) >= 1:
                    seq_len = args[0].shape[-3]
                    # smallest power of two >= seq_len, capped at the default 64
                    chunk_size = 1 << (seq_len - 1).bit_length()
                    chunk_size = min(chunk_size, 64)
                    kwargs["chunk_size"] = chunk_size
                return _orig(*args, **kwargs)

            wrapper._reasonflow_patched = True  # type: ignore[attr-defined]
            module.chunk_gated_delta_rule = wrapper  # type: ignore[attr-defined]

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
        asks_ok = self.asks.score_branch(branch_id, prefill_out.hidden_states)
        if not asks_ok:
            return self.generate_baseline_branch(problem, branch_id)

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

    def _can_batch(self, prefix_pkv: Any) -> bool:
        """Return True iff ``prefix_pkv`` can be expanded and row-selected."""
        if prefix_pkv is None:
            return False
        try:
            get_cache_adapter(prefix_pkv)
        except TypeError:
            return False
        return True

    def generate_batch(
        self,
        problem: str,
        branch_ids: List[int],
        prefix_ids: torch.Tensor,
        prefix_pkv,
        prefix_len: int,
    ) -> List[BranchResult]:
        """Generate many branches with one batched prefill + ASKS partitioning."""
        if not branch_ids:
            return []

        if not self._can_batch(prefix_pkv):
            return [
                self.generate(problem, b, prefix_ids, prefix_pkv, prefix_len)
                for b in branch_ids
            ]

        B = len(branch_ids)
        prefix_str = self.shared_prefix(problem)

        suffix_id_list = [
            self._tokenize("\n" + self.branch_hint(b), add_special_tokens=False)["input_ids"]
            for b in branch_ids
        ]
        suffix_lens = [s.shape[1] for s in suffix_id_list]
        max_L = max(suffix_lens)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 1)
        if pad_token_id is None:
            pad_token_id = 1

        suffix_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long, device=self.device)
        suffix_mask = torch.zeros((B, max_L), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((B, max_L), dtype=torch.long, device=self.device)
        for b, ids in enumerate(suffix_id_list):
            L = ids.shape[1]
            start = max_L - L
            suffix_ids[b, start:] = ids[0]
            suffix_mask[b, start:] = 1
            position_ids[b, start:] = torch.arange(prefix_len, prefix_len + L, device=self.device)

        prefix_mask = torch.ones((B, prefix_len), dtype=torch.long, device=self.device)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        if B == 1:
            expanded_pkv = expand_kv(self.clone_kv_cache(prefix_pkv), B)
        else:
            expanded_pkv = expand_kv(prefix_pkv, B)

        with torch.inference_mode():
            prefill_out = self.model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                past_key_values=expanded_pkv,
                use_cache=True,
                output_hidden_states=True,
                position_ids=position_ids,
            )

        verdicts = self.asks.score_branches(branch_ids, prefill_out.hidden_states)
        approved_idx = [i for i, b in enumerate(branch_ids) if verdicts.get(b, False)]
        rejected_idx = [i for i, b in enumerate(branch_ids) if not verdicts.get(b, False)]

        results: List[Optional[BranchResult]] = [None] * B

        if approved_idx:
            selected_cache = select_kv_cache_rows(prefill_out.past_key_values, approved_idx)
            first_logits = prefill_out.logits[approved_idx, -1, :]

            init_len = prefix_len + max_L
            total_len = init_len + self.config.max_new_tokens
            decode_mask = torch.zeros(
                (len(approved_idx), total_len),
                dtype=torch.long,
                device=self.device,
            )
            decode_mask[:, :init_len] = attention_mask[approved_idx]

            approved_seq_lens = [prefix_len + suffix_lens[i] for i in approved_idx]

            if self.config.max_new_tokens == 0:
                generated_ids_list = [
                    suffix_ids.new_empty((0,), device=self.device) for _ in approved_idx
                ]
                confidences = [0.0] * len(approved_idx)
            else:
                generated_ids_list, confidences = self.decoder.decode_batch(
                    first_logits,
                    selected_cache,
                    decode_mask,
                    self.config.max_new_tokens,
                    approved_seq_lens,
                )

            for j, batch_idx in enumerate(approved_idx):
                branch_id = branch_ids[batch_idx]
                hint = self.branch_hint(branch_id)
                generated_ids = generated_ids_list[j]
                real_suffix_start = max_L - suffix_lens[batch_idx]
                real_suffix = suffix_ids[batch_idx, real_suffix_start:]

                full_sequence = torch.cat(
                    [prefix_ids, real_suffix.unsqueeze(0), generated_ids.unsqueeze(0)], dim=1
                )
                full_text = self.tokenizer.decode(full_sequence[0], skip_special_tokens=True)
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                prompt = prefix_str + "\n" + hint

                results[batch_idx] = BranchResult(
                    branch_id=branch_id,
                    prompt=prompt,
                    text=text,
                    full_text=full_text,
                    generation_confidence=confidences[j],
                )

        for batch_idx in rejected_idx:
            branch_id = branch_ids[batch_idx]
            results[batch_idx] = self.generate_baseline_branch(problem, branch_id)

        return results

    def generate_baseline_branch(self, problem: str, branch_id: int) -> BranchResult:
        """Public alias for baseline branch generation."""
        return self._generate_baseline_branch(problem, branch_id)
