"""Minimal demo: run RKSC on a single problem with a small model."""

import argparse

import torch

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run a single RKSC solve.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Hugging Face model id (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--problem",
        default=(
            "A bakery sells cupcakes in boxes of 6 and 12. "
            "If a customer wants exactly 30 cupcakes, "
            "what is the smallest number of boxes they can buy?"
        ),
    )
    args = parser.parse_args()

    cfg = EngineConfig(
        branching_factor=args.branching_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        device=args.device,
    )

    torch.manual_seed(args.seed)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    engine = MultiBranchEngine(model, tokenizer, cfg)

    print("\n=== RKSC solve ===")
    rksc = engine.solve(args.problem)
    print(f"Best answer:\n{rksc.best_text}\n")
    print(f"Generation: {rksc.generation_time_ms:.1f} ms")
    print(f"Verification: {rksc.verification_time_ms:.1f} ms")
    print(f"Total: {rksc.total_time_ms:.1f} ms")
    print(f"Skipped verification: {rksc.skipped_verification}")
    for b in rksc.branches:
        print(
            f"  branch {b.branch_id}: conf={b.generation_confidence:.3f} "
            f"verify={b.verification_score:.3f} exit_layer={b.early_exit_layer}"
        )

    print("\n=== Baseline solve ===")
    baseline = engine.baseline_solve(args.problem)
    print(f"Best answer:\n{baseline.best_text}\n")
    print(f"Total: {baseline.total_time_ms:.1f} ms")

    speedup = baseline.total_time_ms / max(rksc.total_time_ms, 1e-9)
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
