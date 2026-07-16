"""Compare RKSC against the no-sharing baseline across a few toy problems."""

import argparse
import statistics
import time

import torch

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer

PROBLEMS = [
    "A bakery sells cupcakes in boxes of 6 and 12. "
    "If a customer wants exactly 30 cupcakes, "
    "what is the smallest number of boxes they can buy?",
    "A train 120 meters long crosses a platform 240 meters long in 18 seconds. "
    "What is the speed of the train in meters per second?",
    "A farmer has chickens and rabbits. There are 35 heads and 94 legs in total. "
    "How many chickens and how many rabbits are there?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = EngineConfig(
        branching_factor=args.branching_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        device=args.device,
    )
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    engine = MultiBranchEngine(model, tokenizer, cfg)

    rksc_times = []
    baseline_times = []
    for problem in PROBLEMS:
        print(f"\nProblem: {problem}")
        rksc = engine.solve(problem)
        baseline = engine.baseline_solve(problem)
        print(f"  RKSC total: {rksc.total_time_ms:.1f} ms")
        print(f"  Baseline total: {baseline.total_time_ms:.1f} ms")
        print(f"  Speedup: {baseline.total_time_ms / max(rksc.total_time_ms, 1e-9):.2f}x")
        rksc_times.append(rksc.total_time_ms)
        baseline_times.append(baseline.total_time_ms)
        time.sleep(0.2)

    mean_speedup = statistics.mean(baseline_times) / max(statistics.mean(rksc_times), 1e-9)
    print(f"\nMean RKSC: {statistics.mean(rksc_times):.1f} ms")
    print(f"Mean baseline: {statistics.mean(baseline_times):.1f} ms")
    print(f"Mean speedup: {mean_speedup:.2f}x")


if __name__ == "__main__":
    main()
