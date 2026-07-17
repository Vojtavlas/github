"""Confidence-gated verification of a proposed answer."""

import time
from typing import Optional, Tuple

import torch

from .config import EngineConfig


class Verifier:
    """Score an answer for a problem using a CGEE-gated verifier model."""

    def __init__(self, model, tokenizer, cgee, config: EngineConfig, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.cgee = cgee
        self.config = config
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

    def _tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.device)

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
