"""Confidence-gated verification of a proposed answer."""

import time
from typing import Dict, Optional, Tuple, cast

import torch

from .config import EngineConfig


class Verifier:
    """Score an answer for a problem using a CGEE-gated verifier model."""

    def __init__(self, model, tokenizer, cgee, config: EngineConfig, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cgee = cgee
        self.config = config
        self._yes_ids = self._verifier_token_ids(
            ["YES", "Yes", "yes", " YES", " Yes", " yes"]
        )
        self._no_ids = self._verifier_token_ids(
            ["NO", "No", "no", " NO", " No", " no"]
        )
        if device is not None:
            self.device = device
        elif config.device is not None:
            self.device = config.device
        else:
            self.device = next(model.parameters()).device

    def _verifier_prompt(self, problem: str, answer: str) -> str:
        return (
            f"Problem: {problem}\n\n"
            f"Proposed answer: {answer}\n\n"
            "Is this answer correct? Answer YES or NO."
        )

    def _verifier_token_ids(self, variants):
        """Collect all token IDs produced by a set of string variants."""
        ids = set()
        for text in variants:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.update(encoded)
        return ids

    def _verifier_score(self, logits: torch.Tensor) -> float:
        """Score a verifier prompt, returning a probability-like value."""
        if logits.ndim > 1:
            logits = logits[0]

        if not self._yes_ids and not self._no_ids:
            return 0.5

        probs = torch.softmax(logits, dim=-1)
        yes_mass = probs[list(self._yes_ids)].sum().item() if self._yes_ids else 0.0
        no_mass = probs[list(self._no_ids)].sum().item() if self._no_ids else 0.0
        return yes_mass / (yes_mass + no_mass + 1e-10)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        return cast(
            Dict[str, torch.Tensor],
            self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_seq_len,
            ).to(self.device),
        )

    def verify(self, problem: str, answer: str) -> Tuple[float, Optional[int], float]:
        """Verify an answer and return (score, exit_layer, verify_ms)."""
        prompt = self._verifier_prompt(problem, answer)
        inputs = self._tokenize(prompt)
        t0 = time.perf_counter()
        logits, exit_layer, _ = self.cgee.analyze(
            self.model, inputs["input_ids"], inputs["attention_mask"]
        )
        verify_ms = (time.perf_counter() - t0) * 1000
        score = self._verifier_score(logits[0])
        return score, exit_layer, verify_ms
