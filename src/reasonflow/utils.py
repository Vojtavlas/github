"""Small helpers shared across ReasonFlow modules."""

import copy
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


def clone_kv_cache(pkv):
    """Return a deep copy of a prefix KV cache so branches cannot mutate it."""
    if pkv is None:
        return None
    # Modern Hugging Face Cache objects (DynamicCache and model-specific
    # subclasses such as Qwen3_5DynamicCache) expose key_cache/value_cache.
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        new_cache = copy.copy(pkv)
        new_cache.key_cache = [
            k.clone() if k is not None else None for k in pkv.key_cache
        ]
        new_cache.value_cache = [
            v.clone() if v is not None else None for v in pkv.value_cache
        ]
        return new_cache
    # Older iterable Cache objects.
    if hasattr(pkv, "update") and hasattr(pkv, "__iter__"):
        new_cache = DynamicCache()
        for layer_idx, (key_states, value_states, *_) in enumerate(pkv):
            new_cache.update(
                key_states.clone(),
                value_states.clone(),
                layer_idx=layer_idx,
            )
        return new_cache
    # Legacy tuple cache: tuple of (key, value) tensors per layer.
    return tuple((k.clone(), v.clone()) for k, v in pkv)


def expand_kv(pkv, batch_size: int):
    """Expand a batch-1 KV cache to a target batch size."""
    if pkv is None or batch_size == 1:
        return pkv
    # Native batched expansion is available on modern Hugging Face Cache classes.
    if hasattr(pkv, "batch_repeat_interleave"):
        expanded = clone_kv_cache(pkv)
        expanded.batch_repeat_interleave(batch_size)
        return expanded
    # Cache objects with key_cache/value_cache lists.
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        expanded = copy.copy(pkv)
        expanded.key_cache = [
            k.repeat_interleave(batch_size, dim=0) if k is not None else None
            for k in pkv.key_cache
        ]
        expanded.value_cache = [
            v.repeat_interleave(batch_size, dim=0) if v is not None else None
            for v in pkv.value_cache
        ]
        return expanded
    # Fallback for legacy tuple caches.
    new_cache = DynamicCache()
    if hasattr(pkv, "__iter__"):
        for layer_idx, (key_states, value_states, *_) in enumerate(pkv):
            new_cache.update(
                key_states.repeat_interleave(batch_size, dim=0),
                value_states.repeat_interleave(batch_size, dim=0),
                layer_idx=layer_idx,
            )
        return new_cache
    return pkv
