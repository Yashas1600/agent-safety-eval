"""CLI entry point for agent-safety-eval."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ase", description="Agent Safety Eval harness"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available tasks")

    run_p = sub.add_parser("run", help="Run evaluation")
    run_p.add_argument(
        "--model",
        action="append",
        required=True,
        help="model spec (provider:id), e.g. mock:careful, gemini:gemini-2.5-flash",
    )
    run_p.add_argument("--suite", default="v1", help="Task suite name")
    run_p.add_argument("--seeds", type=int, default=3, help="Seeds per task")
    run_p.add_argument(
        "--concurrency", type=int, default=4, help="Parallel episodes"
    )
    run_p.add_argument(
        "--transport",
        choices=["in-process", "stdio"],
        default="in-process",
    )
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

    if args.command == "list":
        _cmd_list()
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "replay":
        print("Replay not yet implemented.")
        sys.exit(1)


# ── list ──────────────────────────────────────────────────────────


def _cmd_list() -> None:
    """List all tasks in the default suite."""
    from agent_safety_eval.tasks.schema import load_suite

    tasks = load_suite("suite_v1")

    # Try rich table, fall back to plain
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="suite_v1 tasks")
        table.add_column("ID", style="bold")
        table.add_column("Category")
        table.add_column("Description")
        for t in tasks:
            # First line of instruction as description
            desc = t.instruction.split("\n")[0][:80]
            table.add_row(t.id, t.category, desc)
        console.print(table)
    except ImportError:
        print(f"{'ID':<45} {'Category':<22} Description")
        print("-" * 100)
        for t in tasks:
            desc = t.instruction.split("\n")[0][:50]
            print(f"{t.id:<45} {t.category:<22} {desc}")

    print(f"\n{len(tasks)} tasks total.")


# ── run ───────────────────────────────────────────────────────────


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a model spec string like 'provider:model_id'.

    Returns (provider, model_id). For mock adapters, model_id is the policy.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid model spec: {spec!r}. Expected format: provider:id"
        )
    provider, model_id = spec.split(":", 1)
    return provider, model_id


def _get_adapter(provider: str, model_id: str) -> Any:
    """Instantiate a model adapter given provider and model_id."""
    if provider == "mock":
        from agent_safety_eval.adapters.mock import get_mock_adapter

        return get_mock_adapter(model_id)

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Available: mock, gemini, anthropic, openai, adk"
    )


def _cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation for all specified models against the suite."""
    from agent_safety_eval.parallel import (
        EpisodeSpec,
        run_parallel,
    )
    from agent_safety_eval.report import (
        episode_result_to_dict,
        generate_csv,
        generate_json,
        generate_markdown,
        print_comparison_table,
    )
    from agent_safety_eval.runner import run_episode
    from agent_safety_eval.tasks.schema import load_suite

    suite_name = f"suite_{args.suite}" if not args.suite.startswith("suite_") else args.suite
    tasks = load_suite(suite_name)
    print(f"Loaded {len(tasks)} tasks from {suite_name}.")

    # Parse model specs
    model_specs = [_parse_model_spec(s) for s in args.model]

    if args.dry_run:
        print("Dry run. Planned episodes:")
        for provider, model_id in model_specs:
            n = len(tasks) * args.seeds
            print(f"  {provider}:{model_id}: {n} episodes ({len(tasks)} tasks x {args.seeds} seeds)")
        return

    # Build episode specs
    all_specs: list[EpisodeSpec] = []
    spec_meta: list[tuple[str, str, Any]] = []  # (adapter_name, task_id, task)
    idx = 0
    for provider, model_id in model_specs:
        adapter_name = f"{provider}:{model_id}"
        for task in tasks:
            for seed in range(args.seeds):
                all_specs.append(
                    EpisodeSpec(
                        index=idx,
                        task_id=task.id,
                        provider=provider,
                        seed=seed,
                        config={
                            "model_id": model_id,
                            "adapter_name": adapter_name,
                        },
                    )
                )
                spec_meta.append((adapter_name, task.id, task))
                idx += 1

    print(f"Running {len(all_specs)} episodes ({args.concurrency} concurrent)...")

    # Build the episode callable for the parallel runner
    # We need a lookup from task_id to task and a way to create adapters
    task_by_id = {t.id: t for t in tasks}

    async def episode_fn(spec: EpisodeSpec) -> Any:
        """Run a single episode (called by the parallel runner)."""
        from agent_safety_eval.parallel import EpisodeResult as ParallelResult

        adapter_name = spec.config["adapter_name"]
        model_id = spec.config["model_id"]
        task = task_by_id[spec.task_id]
        adapter = _get_adapter(spec.provider, model_id)

        result = run_episode(task, adapter, adapter_name)

        return ParallelResult(
            index=spec.index,
            task_id=spec.task_id,
            provider=spec.provider,
            seed=spec.seed,
            success=True,
            data=result,
        )

    # Run
    parallel_results, stats = asyncio.run(
        run_parallel(
            all_specs,
            episode_fn,
            concurrency=args.concurrency,
        )
    )

    # Group results by adapter name
    all_results: dict[str, list[dict[str, Any]]] = {}
    for pr, (adapter_name, _, _) in zip(parallel_results, spec_meta):
        if pr.success and pr.data is not None:
            rd = episode_result_to_dict(pr.data)
        else:
            # Failed episode
            rd = {
                "task_id": pr.task_id,
                "adapter_name": adapter_name,
                "category": "unknown",
                "success": False,
                "wrong_tool_calls": [],
                "forbidden_violations": [],
                "unconfirmed_irreversible": [],
                "over_refusal": False,
                "injection_followed": False,
                "turns_used": 0,
                "error": pr.error,
            }
        all_results.setdefault(adapter_name, []).append(rd)

    run_stats_dict = {
        "total_episodes": stats.total_episodes,
        "total_duration_sec": round(stats.total_duration_sec, 3),
        "episodes_per_sec": round(stats.episodes_per_sec, 1),
        "p50_latency_sec": round(stats.p50_latency_sec, 4),
        "p95_latency_sec": round(stats.p95_latency_sec, 4),
        "total_tokens": stats.total_tokens,
    }

    # Save results
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw episode results as JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(run_stats_dict, f, indent=2)

    print(f"Results saved to {out_dir}/")
    print()

    # Print summary table
    print_comparison_table(all_results, run_stats_dict)


# ── report ────────────────────────────────────────────────────────


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate reports from a results directory."""
    from agent_safety_eval.report import (
        generate_csv,
        generate_json,
        generate_markdown,
        print_comparison_table,
    )

    results_dir = Path(args.results_dir)
    results_file = results_dir / "results.json"
    stats_file = results_dir / "stats.json"

    if not results_file.exists():
        print(f"Error: {results_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(results_file) as f:
        all_results = json.load(f)

    run_stats = None
    if stats_file.exists():
        with open(stats_file) as f:
            run_stats = json.load(f)

    # Generate all report formats
    md = generate_markdown(all_results, run_stats)
    with open(results_dir / "report.md", "w") as f:
        f.write(md)

    json_report = generate_json(all_results, run_stats)
    with open(results_dir / "report.json", "w") as f:
        f.write(json_report)

    csv_report = generate_csv(all_results)
    with open(results_dir / "report.csv", "w") as f:
        f.write(csv_report)

    print(f"Reports generated in {results_dir}/:")
    print(f"  report.md   (markdown summary)")
    print(f"  report.json (full JSON with CIs)")
    print(f"  report.csv  (per-episode CSV)")
    print()

    # Also print the comparison table
    print_comparison_table(all_results, run_stats)


if __name__ == "__main__":
    main()
