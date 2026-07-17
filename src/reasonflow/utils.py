"""Small helpers shared across ReasonFlow modules."""

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[str] = None,
    attn_impl: str = "sdpa",
) -> tuple:
    """Load a causal LM and its tokenizer with sensible defaults."""
    tok: Any = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            raise ValueError(
                "Tokenizer has no pad_token or eos_token; cannot set pad_token."
            )
    tok.padding_side = "left"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = str(device)
    dtype = torch.bfloat16 if device_str.startswith("cuda") else torch.float32

    kwargs = {
        "dtype": dtype,
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    if device_str.startswith("cuda"):
        kwargs["device_map"] = "auto" if device_str == "cuda" else device_str
    else:
        kwargs["low_cpu_mem_usage"] = True

    model: Any = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not kwargs.get("device_map"):
        model = model.to(device_str)
    model.eval()
    return model, tok


def squeeze_hidden(h):
    """Return the last-token hidden state for 1-D, 2-D, or 3-D inputs."""
    if isinstance(h, tuple):
        h = h[0]
    if h.dim() == 1:
        return h
    if h.dim() == 2:
        return h[-1, :]
    if h.dim() == 3:
        last = h[:, -1, :]
        if last.shape[0] == 1:
            return last[0]
        raise ValueError(
            f"Expected batch size 1 for 3-D hidden tensor, got {last.shape[0]}."
        )
    raise ValueError(f"Unsupported hidden tensor dimension: {h.dim()}.")
