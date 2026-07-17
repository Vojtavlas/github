"""Next-token sampling strategies."""

from typing import Tuple

import torch


class Sampler:
    """Sample the next token from logits using temperature, top-p, or greedy decoding."""

    def __init__(self, config):
        self.config = config

    def sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the sampled token id and its probability."""
        if self.config.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.config.temperature}"
            )

        top_p = self.config.top_p
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        if self.config.temperature == 0:
            probs = torch.softmax(logits, dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            confidence = probs.gather(-1, next_token).squeeze(-1)
            return next_token.squeeze(-1), confidence

        temperature = self.config.temperature
        min_temperature = 1e-5
        if temperature < min_temperature:
            temperature = min_temperature
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            remove = cumsum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            filtered = probs.scatter(
                -1, sorted_indices, torch.where(remove, 0.0, sorted_probs)
            )
            filtered_sum = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            probs = filtered / filtered_sum

        next_token = torch.multinomial(probs, num_samples=1)
        confidence = probs.gather(-1, next_token).squeeze(-1)
        return next_token.squeeze(-1), confidence
