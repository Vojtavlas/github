"""Autoregressive decoding helpers."""

from typing import List, Tuple

import torch


class Decoder:
    """Generate tokens autoregressively given a model and sampler."""

    def __init__(self, model, tokenizer, sampler, config):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.config = config

    def decode(
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

        if first_input_ids.size(0) != 1:
            raise ValueError(
                f"Decoder currently only supports batch_size == 1, got {first_input_ids.size(0)}"
            )

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
            next_token, conf = self.sampler.sample(logits)
            generated.append(next_token)
            confidences.append(conf)
            if next_token.item() == eos_id:
                finished = True
                break
            curr_ids = next_token.unsqueeze(-1)
            curr_mask = torch.cat(
                [
                    curr_mask,
                    torch.ones(
                        (curr_mask.size(0), 1),
                        dtype=curr_mask.dtype,
                        device=curr_mask.device,
                    ),
                ],
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

    def continue_generate(
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

        if first_logits.size(0) != 1:
            raise ValueError(
                f"Decoder currently only supports batch_size == 1, got {first_logits.size(0)}"
            )

        next_token, conf = self.sampler.sample(first_logits)
        generated.append(next_token)
        confidences.append(conf)
        if next_token.item() == eos_id or max_new_tokens <= 1:
            generated_ids = torch.stack(generated, dim=1)
            mean_conf = torch.stack(confidences).mean().item()
            return generated_ids, mean_conf, pkv

        curr_ids = next_token.unsqueeze(-1)
        curr_mask = torch.cat(
            [
                curr_mask,
                torch.ones(
                    (curr_mask.size(0), 1),
                    dtype=curr_mask.dtype,
                    device=curr_mask.device,
                ),
            ],
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
            next_token, conf = self.sampler.sample(logits)
            generated.append(next_token)
            confidences.append(conf)
            if next_token.item() == eos_id:
                break
            curr_ids = next_token.unsqueeze(-1)
            curr_mask = torch.cat(
                [
                    curr_mask,
                    torch.ones(
                        (curr_mask.size(0), 1),
                        dtype=curr_mask.dtype,
                        device=curr_mask.device,
                    ),
                ],
                dim=1,
            )
            pkv = out.past_key_values

        generated_ids = torch.stack(generated, dim=1)
        mean_conf = torch.stack(confidences).mean().item()
        return generated_ids, mean_conf, pkv
