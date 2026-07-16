"""Small helpers shared across ReasonFlow modules."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


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
        "torch_dtype": dtype,
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

    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = False

    return model, tok


def get_transformer_layers(model) -> List[torch.nn.Module]:
    """Locate the list of transformer decoder layers for hooking."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)

    for name, module in model.named_modules():
        if name.count(".") == 2 and ("layers" in name or name.endswith(".h")):
            if isinstance(module, torch.nn.ModuleList):
                return list(module)
    raise ValueError("Could not locate transformer layers for CGEE hooks")


def squeeze_hidden(h):
    """Return the last-token hidden state, squeezing a leading batch dim of 1."""
    if isinstance(h, tuple):
        h = h[0]
    if h.dim() == 3:
        return h[:, -1, :].squeeze(0)
    return h.squeeze(0)


def expand_kv(pkv, batch_size: int):
    """Expand a batch-1 KV cache to a target batch size."""
    if pkv is None or batch_size == 1:
        return pkv
    new_cache = DynamicCache()
    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
        for li in range(len(pkv.key_cache)):
            new_cache.update(
                pkv.key_cache[li].repeat_interleave(batch_size, dim=0),
                pkv.value_cache[li].repeat_interleave(batch_size, dim=0),
                layer_idx=li,
            )
        return new_cache
    if hasattr(pkv, "layers") and len(pkv.layers) > 0:
        for li, layer in enumerate(pkv.layers):
            new_cache.update(
                layer.key.repeat_interleave(batch_size, dim=0),
                layer.value.repeat_interleave(batch_size, dim=0),
                layer_idx=li,
            )
        return new_cache
    return pkv
