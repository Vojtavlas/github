"""Run a small accuracy/speedup evaluation on a reasoning dataset."""

import argparse

from reasonflow import EngineConfig, MultiBranchEngine, load_model_and_tokenizer
from reasonflow.eval import EvalConfig as EvalConfigEval
from reasonflow.eval import Evaluator, HFTextDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReasonFlow on a reasoning dataset.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--metric", default="numeric_match")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default="eval_report.json")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    cfg = EngineConfig(
        branching_factor=args.branching_factor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    engine = MultiBranchEngine(model, tokenizer, cfg)

    eval_cfg = EvalConfigEval(
        max_problems=args.max_problems,
        metric=args.metric,
        split=args.split,
    )
    dataset = HFTextDataset.from_name(
        args.dataset,
        config=args.dataset_config,
        split=args.split,
        problem_column="question",
        answer_column="answer",
        max_problems=args.max_problems,
    )

    evaluator = Evaluator(engine, eval_cfg)
    report = evaluator.run(dataset)
    print(f"Accuracy:     {report.accuracy:.3f}")
    print(f"Baseline acc: {report.baseline_accuracy:.3f}")
    print(f"Speedup:      {report.speedup:.2f}x")
    print(f"RKSC ms:      {report.rksc_ms:.1f}")
    print(f"Baseline ms:  {report.baseline_ms:.1f}")

    if args.output_json:
        report.save_json(args.output_json)
        print(f"Saved JSON report to {args.output_json}")
    if args.output_csv:
        report.save_csv(args.output_csv)
        print(f"Saved CSV report to {args.output_csv}")


if __name__ == "__main__":
    main()
