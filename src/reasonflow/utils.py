"""Small helpers shared across ReasonFlow modules."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[str] = None,
    attn_impl: str = "sdpa",
) -> tuple:
    """Load a causal LM and its tokenizer with sensible defaults."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

    kwargs = {
        "dtype": dtype,
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    if "cuda" in str(device):
        kwargs["device_map"] = "cuda:0"
    else:
        kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not kwargs.get("device_map"):
        model = model.to(device)
    model.eval()
    return model, tok


def squeeze_hidden(h):
    """Return the last-token hidden state, squeezing a leading batch dim of 1."""
    if isinstance(h, tuple):
        h = h[0]
    if h.dim() == 3:
        return h[:, -1, :].squeeze(0)
    return h.squeeze(0)
