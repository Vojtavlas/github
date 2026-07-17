# ReasonFlow domain glossary

This file gives names to the concepts and seams in ReasonFlow. Use it when discussing architecture, writing tests, or naming new modules.

## Core domain

- **Problem** — The input math/word problem string passed to `MultiBranchEngine.solve()`.
- **Shared prefix** — The common prompt text computed once per problem: `"You are a helpful assistant...\n\nProblem: {problem}\n\nReasoning:"`.
- **Branch hint** — One of `DEFAULT_BRANCH_HINTS` appended to the shared prefix to encourage a different reasoning strategy per branch.
- **Branch** — A single generated reasoning path, including its prompt, generated text, confidence, and verification state.

## Modules and seams

- **ASKSManager** — State module that captures root hidden states and KV cache, then gates whether a branch may reuse the root prefix KV cache.
  - **SimilarityMetric** — Pluggable seam for computing similarity between root and branch hidden states (e.g. `CosineSimilarity`).
  - **WeightingStrategy** — Pluggable seam for combining per-layer similarity scores (e.g. `ExponentialWeights`).
- **BranchGenerator** — Module that builds a branch prompt, runs the prefix+hint prefill, checks ASKS, and decodes. `generate_baseline_branch()` generates from scratch without KV reuse.
- **Decoder** — Module that runs autoregressive token generation given a starting input or the first-token logits.
- **Sampler** — Module that samples one token and its probability from logits, supporting temperature, top-p, and greedy decoding.
- **CGEEAnalyzer** — Module that implements two-level confidence-gated early exit:
  - Level 1: skip verification when one branch is confident and leads by a wide gap.
  - Level 2: exit the verifier forward pass early when per-layer entropy becomes low and stable.
  - **EntropyTracker** — Pure module that computes and records per-layer output entropy.
  - **EarlyExitStrategy** — Pure module that decides when the entropy curve justifies early exit.
  - **HookAdapter** — Adapter that bridges PyTorch forward hooks to the `CGEEAnalyzer` callback.
- **Verifier** — Module that prompts the model with `"Is this answer correct? Answer YES or NO."` and scores the answer by the probability of `YES`.
- **RSBCMManager** — Block cache manager that allocates and evicts KV blocks by `importance / (tree_depth + 1)` priority.
- **CachePort / CacheAdapter** — Seam and adapter set for cloning and expanding different Hugging Face KV cache formats (`DynamicCache`, model-specific subclasses, iterable caches, legacy tuples).
- **ModelAdapter** — Adapter seam that locates transformer decoder layers for different model architectures (LLaMA, GPT-2, GPT-NeoX, heuristic fallback).
- **MultiBranchEngine** — Thin coordinator module that instantiates the modules above and orchestrates `solve()` and `baseline_solve()`.

## Result types

- **BranchResult** — Dataclass for a single branch: prompt, generated text, confidence, verification score, and early-exit layer.
- **SolveResult** — Aggregated result for a problem: best branch text, all branches, timing, and whether verification was skipped.

## Configuration

- **EngineConfig** — Top-level configuration for a `MultiBranchEngine`.
- **RKSCConfig** — Hyper-parameters for ASKS and CGEE.
- **RSBCMConfig** — Capacity control for the RSBCM block cache.
