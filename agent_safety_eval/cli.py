"""CLI entry point for agent-safety-eval."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="ase", description="Agent Safety Eval harness")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available tasks")

    run_p = sub.add_parser("run", help="Run evaluation")
    run_p.add_argument("--model", action="append", required=True, help="model spec (provider:id)")
    run_p.add_argument("--suite", default="v1", help="Task suite name")
    run_p.add_argument("--seeds", type=int, default=3, help="Seeds per task")
    run_p.add_argument("--concurrency", type=int, default=4, help="Parallel episodes")
    run_p.add_argument("--transport", choices=["in-process", "stdio"], default="in-process")
    run_p.add_argument("--out", required=True, help="Output directory")
    run_p.add_argument("--dry-run", action="store_true")

    report_p = sub.add_parser("report", help="Generate report from results")
    report_p.add_argument("results_dir", help="Path to results directory")

    replay_p = sub.add_parser("replay", help="Replay a transcript")
    replay_p.add_argument("transcript", help="Path to transcript JSONL")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
