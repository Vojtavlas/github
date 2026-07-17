"""Compare RKSC against the no-sharing baseline across several problems.

This demo loads a model once, warms up the GPU, then times both ``solve``
(RKSC with prefix sharing and CGEE gating) and ``baseline_solve`` over a
small set of problems.  ``torch.cuda.synchronize()`` is used around each
timed call so the measurement reflects wall-clock GPU time.
"""

import argparse
import json
import statistics
import time

import torch

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer

PROBLEMS = [
    (
        "cupcakes",
        "A bakery sells cupcakes in boxes of 6 and 12. "
        "If a customer wants exactly 30 cupcakes, "
        "what is the smallest number of boxes they can buy?",
    ),
    (
        "train",
        "A train 120 meters long crosses a platform 240 meters long in 18 seconds. "
        "What is the speed of the train in meters per second?",
    ),
    (
        "chickens",
        "A farmer has chickens and rabbits. There are 35 heads and 94 legs in total. "
        "How many chickens and how many rabbits are there?",
    ),
    (
        "triangle",
        "The sides of a triangle are 5, 12, and 13. "
        "What is the area of the triangle?",
    ),
    (
        "average",
        "The average of five numbers is 24. The first four numbers are 18, 22, 28, and 30. "
        "What is the fifth number?",
    ),
]


def _sync_device(device):
    """Synchronize the compute device before taking wall-clock timings."""
    if device is not None and device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed(engine, method_name: str, problem: str, device):
    """Run ``engine.solve`` or ``engine.baseline_solve`` with precise timing."""
    fn = getattr(engine, method_name)
    _sync_device(device)
    t0 = time.perf_counter()
    result = fn(problem)
    _sync_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def _run_problem(engine, problem: str, device: str, order: str):
    """Return (rksc_ms, baseline_ms) for one problem with the given call order."""
    if order == "rksc_first":
        _, rksc_ms = _timed(engine, "solve", problem, device)
        _, baseline_ms = _timed(engine, "baseline_solve", problem, device)
    else:
        _, baseline_ms = _timed(engine, "baseline_solve", problem, device)
        _, rksc_ms = _timed(engine, "solve", problem, device)
    return rksc_ms, baseline_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RKSC against the no-sharing baseline."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="JSON file to write results")
    args = parser.parse_args()

    cfg = EngineConfig(
        branching_factor=args.branching_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    engine = MultiBranchEngine(model, tokenizer, cfg)

    # Determine the actual device for synchronization.
    device = args.device or str(next(model.parameters()).device)

    # Warm-up runs to exclude compile / cache warm-up from the timed results.
    print(
        f"Warming up with {args.warmup} run(s) "
        f"(max_new_tokens={args.max_new_tokens}, branching_factor={args.branching_factor}) ..."
    )
    for i in range(args.warmup):
        torch.manual_seed(args.seed + i)
        for _, problem in PROBLEMS:
            engine.solve(problem)
            engine.baseline_solve(problem)

    per_problem_records = []
    for tag, problem in PROBLEMS:
        rksc_times = []
        baseline_times = []
        for run in range(args.runs):
            # Alternate order per run to average out any cache-order bias.
            order = "rksc_first" if run % 2 == 0 else "baseline_first"
            torch.manual_seed(args.seed + args.warmup + run)
            rksc_ms, baseline_ms = _run_problem(engine, problem, device, order)
            rksc_times.append(rksc_ms)
            baseline_times.append(baseline_ms)

        mean_rksc = statistics.mean(rksc_times)
        mean_baseline = statistics.mean(baseline_times)
        speedup = mean_baseline / mean_rksc if mean_rksc > 0 else float("nan")
        per_problem_records.append(
            {
                "tag": tag,
                "problem": problem,
                "rksc_ms": mean_rksc,
                "baseline_ms": mean_baseline,
                "speedup": speedup,
                "rksc_times": rksc_times,
                "baseline_times": baseline_times,
            }
        )
        print(
            f"{tag:12}  RKSC {mean_rksc:8.1f} ms  "
            f"Baseline {mean_baseline:8.1f} ms  "
            f"Speedup {speedup:5.2f}x"
        )

    overall_rksc = statistics.mean(r["rksc_ms"] for r in per_problem_records)
    overall_baseline = statistics.mean(r["baseline_ms"] for r in per_problem_records)
    overall_speedup = overall_baseline / overall_rksc if overall_rksc > 0 else float("nan")

    print("-" * 70)
    print(
        f"{'OVERALL':12}  RKSC {overall_rksc:8.1f} ms  "
        f"Baseline {overall_baseline:8.1f} ms  "
        f"Speedup {overall_speedup:5.2f}x"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "config": {
                        "model": args.model,
                        "device": device,
                        "branching_factor": args.branching_factor,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "warmup": args.warmup,
                        "runs": args.runs,
                        "seed": args.seed,
                    },
                    "per_problem": per_problem_records,
                    "overall": {
                        "rksc_ms": overall_rksc,
                        "baseline_ms": overall_baseline,
                        "speedup": overall_speedup,
                    },
                },
                f,
                indent=2,
            )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
