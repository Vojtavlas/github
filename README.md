# ReasonFlow — Fast Multi-Branch LLM Reasoning

A small, reproducible research package that accelerates multi-branch LLM
reasoning (self-consistency / tree-of-thought) by combining three ideas from
recent inference-efficiency literature:

1. **Prefix KV sharing** — compute the shared problem prefix once and reuse it
   for every reasoning branch.
2. **ASKS-style hidden-state gating** *(module included)* — decide when a branch
   prompt is semantically close enough to the root to reuse its KV cache.
3. **CGEE** — skip or early-exit the verification forward pass when the model is
   already confident.

The implementation is inspired by **RKSC: Reasoning-Aware KV Cache Sharing and
Confident Early Exit for Multi-Step LLM Inference** (arXiv:2606.09937, June 2026),
which reports ~3× speedup on 7B–10B models with a 0.37% error rate.

## Why this is useful for an internship

- It touches a hot systems/research topic: **inference-time scaling** and
  **efficient LLM inference**.
- It builds on a paper from 2026, showing you follow the latest literature.
- It produces measurable speedup numbers, tests, and a clean codebase that
  interviewers can read and run.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
python examples/simple_demo.py
```

This loads the small `Qwen/Qwen3.5-0.8B` model and compares an RKSC-style solve
against the naive baseline (generate each branch independently + full
verification).

> **Note:** `Qwen/Qwen3.5-0.8B` uses `trust_remote_code=True` and falls back to a
> pure-Pytorch attention path unless `fla`/`causal-conv1d` are installed. Set
> `HF_TOKEN` for higher Hugging Face Hub rate limits. The measured wall-clock
> speedup is workload-dependent and usually larger on bigger models/GPUs; the
> 0.8B CPU demo is intended to show the method is runnable end-to-end.

## What the code does

```python
from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3.5-0.8B")
cfg = EngineConfig(branching_factor=2, max_new_tokens=32)
engine = MultiBranchEngine(model, tokenizer, cfg)

result = engine.solve("A train 120 m long crosses a pole in 6 s. What is its speed?")
print(result.best_text)
print(f"Speedup vs baseline: {engine.baseline_solve(result.problem).total_time_ms / result.total_time_ms:.2f}x")
```

### Core components

- `reasonflow.asks.ASKSManager` — hidden-state cosine-similarity gate for KV reuse.
- `reasonflow.cgee.CGEEAnalyzer` — generation-confidence skip + layer-wise entropy
  early exit via forward hooks.
- `reasonflow.cache.RSBCMManager` — score/depth KV block eviction for deep tree
  search.
- `reasonflow.engine.MultiBranchEngine` — end-to-end branch generation and
  verification.
- `reasonflow.eval.Evaluator` — accuracy/speedup evaluation on reasoning datasets
  with answer extraction and pluggable metrics.

### Evaluating on a dataset

```python
from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer
from reasonflow.eval import EvalConfig, Evaluator, HFTextDataset

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3.5-0.8B")
cfg = EngineConfig(branching_factor=2, max_new_tokens=32)
engine = MultiBranchEngine(model, tokenizer, cfg)

eval_cfg = EvalConfig(max_problems=50, metric="numeric_match")
dataset = HFTextDataset.from_name("openai/gsm8k", split="test", max_problems=50)
evaluator = Evaluator(engine, eval_cfg)
report = evaluator.run(dataset)
print(f"Accuracy: {report.accuracy:.3f}, Speedup: {report.speedup:.2f}x")
report.save_json("eval_report.json")
```

Or run the CLI demo:

```bash
py -3.11 examples/eval_demo.py --max-problems 10 --metric numeric_match
```

## Math snapshot

**ASKS similarity** between a branch and the root:

```
σ_b = Σ_l w_l · (h_b^(l) · h_root^(l)) / (||h_b^(l)|| ||h_root^(l)||)
```

Weights `w_l` are exponentially tilted toward later layers.

**CGEE Level 1** skips verification when:

```
max_b p^(b) ≥ τ_conf   and   (max_b p^(b) - 2ndmax_b p^(b)) / max_b p^(b) ≥ r_gap
```

**CGEE Level 2** exits the verifier forward at layer `l*` when:

```
l* ≥ l_min,   H(l*) < θ,   |H(l*) - H(l*-1)| < ε
```

## Project structure

```
src/reasonflow/
  __init__.py
  config.py
  utils.py
  asks.py
  cgee.py
  cache.py
  cache_adapter.py
  model_adapter.py
  branch_generator.py
  decoder.py
  sampler.py
  verifier.py
  engine.py
  eval.py
  results.py
  metrics.py
examples/
  simple_demo.py
  benchmark_demo.py
  eval_demo.py
tests/
  test_*.py
```

## Tests

```bash
pytest -q
```

Engine integration tests are skipped in CI; set `SKIP_ENGINE_TESTS=0` to run
locally with a small model.

## License

MIT
