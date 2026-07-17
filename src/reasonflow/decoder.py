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

    def _make_mask_buffer(self, attention_mask: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Preallocate a mask buffer sized for the full decode sequence."""
        batch_size = attention_mask.size(0)
        init_len = attention_mask.size(1)
        total_len = init_len + max_new_tokens
        buffer = torch.ones(
            (batch_size, total_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        buffer[:, :init_len] = attention_mask
        return buffer

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
        curr_ids = first_input_ids
        pkv = past_key_values
        eos_id = self.tokenizer.eos_token_id

        if first_input_ids.size(0) != 1:
            raise ValueError(
                f"Decoder currently only supports batch_size == 1, got {first_input_ids.size(0)}"
            )

        mask_buffer = self._make_mask_buffer(attention_mask, max_new_tokens)
        curr_len = first_input_ids.size(1)

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                out = self.model(
                    input_ids=curr_ids,
                    attention_mask=mask_buffer[:, :curr_len],
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
            curr_len += 1
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
        pkv = past_key_values
        eos_id = self.tokenizer.eos_token_id

        if first_logits.size(0) != 1:
            raise ValueError(
                f"Decoder currently only supports batch_size == 1, got {first_logits.size(0)}"
            )

        mask_buffer = self._make_mask_buffer(attention_mask, max_new_tokens)
        curr_len = attention_mask.size(1)

        next_token, conf = self.sampler.sample(first_logits)
        generated.append(next_token)
        confidences.append(conf)
        if next_token.item() == eos_id or max_new_tokens <= 1:
            generated_ids = torch.stack(generated, dim=1)
            mean_conf = torch.stack(confidences).mean().item()
            return generated_ids, mean_conf, pkv

        curr_ids = next_token.unsqueeze(-1)
        curr_len += 1

        for _ in range(max_new_tokens - 1):
            with torch.inference_mode():
                out = self.model(
                    input_ids=curr_ids,
                    attention_mask=mask_buffer[:, :curr_len],
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
            curr_len += 1
            pkv = out.past_key_values

        generated_ids = torch.stack(generated, dim=1)
        mean_conf = torch.stack(confidences).mean().item()
        return generated_ids, mean_conf, pkv

    def decode_batch(
        self,
        first_logits: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        seq_lens: List[int],
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Batched autoregressive decode where each row can finish independently.

        Parameters
        ----------
        first_logits:
            Logits for the first token to sample, shape ``[B, vocab]`` or
            ``[B, 1, vocab]``.
        past_key_values:
            KV cache already prefilled for each batch row.
        attention_mask:
            Preallocated mask of shape ``[B, init_len + max_new_tokens]`` where
            ``init_len = attention_mask.size(1) - max_new_tokens``. The initial
            ``init_len`` columns encode the prefix + suffix; generation columns
            start at index ``init_len``.
        max_new_tokens:
            Maximum number of new tokens to generate per row.
        seq_lens:
            Real sequence length (prefix + suffix) for each row, used to set
            absolute position ids during generation.

        Returns
        -------
        A list of generated token-id tensors (one 1-D tensor per row) and a list
        of per-row mean confidences.
        """
        if first_logits.dim() == 3:
            first_logits = first_logits[:, -1, :]

        batch_size = first_logits.size(0)
        device = first_logits.device
        init_len = attention_mask.size(1) - max_new_tokens
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 1

        active = [True] * batch_size
        generated: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        confidences: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        next_tokens, next_confs = self.sampler.sample(first_logits)
        current_ids = torch.full((batch_size, 1), pad_id, dtype=torch.long, device=device)

        for b in range(batch_size):
            generated[b].append(next_tokens[b])
            confidences[b].append(next_confs[b])
            if next_tokens[b].item() == eos_id:
                active[b] = False
            else:
                current_ids[b, 0] = next_tokens[b]

        if max_new_tokens > 1 and any(active):
            for step in range(1, max_new_tokens):
                if not any(active):
                    break

                active_indices = [i for i, a in enumerate(active) if a]
                new_col = init_len + step - 1
                attention_mask[active_indices, new_col] = 1

                position_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
                position_ids[active_indices] = torch.tensor(
                    [seq_lens[b] + step - 1 for b in active_indices],
                    dtype=torch.long,
                    device=device,
                )

                with torch.inference_mode():
                    out = self.model(
                        input_ids=current_ids,
                        attention_mask=attention_mask[:, : init_len + step],
                        past_key_values=past_key_values,
                        position_ids=position_ids.unsqueeze(1),
                        use_cache=True,
                    )

                logits = out.logits[:, -1, :]
                new_tokens, new_confs = self.sampler.sample(logits)
                past_key_values = out.past_key_values

                for b in range(batch_size):
                    if active[b]:
                        generated[b].append(new_tokens[b])
                        confidences[b].append(new_confs[b])
                        if new_tokens[b].item() == eos_id:
                            active[b] = False
                            current_ids[b, 0] = pad_id
                        else:
                            current_ids[b, 0] = new_tokens[b]
                    else:
                        current_ids[b, 0] = pad_id

        sequences: List[torch.Tensor] = []
        mean_confidences: List[float] = []
        for g, c in zip(generated, confidences):
            if g:
                sequences.append(torch.stack(g, dim=0))
                mean_confidences.append(torch.stack(c).mean().item())
            else:
                sequences.append(torch.empty(0, dtype=torch.long, device=device))
                mean_confidences.append(0.0)

        return sequences, mean_confidences
