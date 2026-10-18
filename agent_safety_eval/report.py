"""Report generation: markdown, JSON, and CSV summaries.

Produces per-category breakdowns and Wilson 95% confidence intervals
for all headline metrics.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from typing import Any


# Wilson 95% CI: p +/- z*sqrt(p*(1-p)/n), clamped to [0,1]
_Z = 1.96


@dataclass
class MetricWithCI:
    """A metric value with a Wilson 95% confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:.1%} [{self.ci_low:.1%}, {self.ci_high:.1%}] (n={self.n})"


def wilson_ci(successes: int, total: int) -> MetricWithCI:
    """Compute a proportion with Wilson 95% confidence interval."""
    if total == 0:
        return MetricWithCI(value=0.0, ci_low=0.0, ci_high=0.0, n=0)
    p = successes / total
    margin = _Z * math.sqrt(p * (1 - p) / total)
    lo = max(0.0, p - margin)
    hi = min(1.0, p + margin)
    return MetricWithCI(value=p, ci_low=lo, ci_high=hi, n=total)


def _count_metric(
    results: list[dict[str, Any]], key: str, invert: bool = False
) -> tuple[int, int]:
    """Count how many results have a truthy value for key.

    If invert is True, count those with falsy value instead.
    """
    total = len(results)
    if invert:
        hits = sum(1 for r in results if not r.get(key))
    else:
        hits = sum(1 for r in results if r.get(key))
    return hits, total


def compute_category_metrics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, MetricWithCI]]:
    """Compute per-category metrics from a flat list of episode result dicts.

    Each result dict should have keys matching EpisodeResult fields:
    task_id, category, success, wrong_tool_calls, unconfirmed_irreversible,
    over_refusal, injection_followed.
    """
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    metrics: dict[str, dict[str, MetricWithCI]] = {}
    for cat, cat_results in sorted(by_cat.items()):
        metrics[cat] = _metrics_for_group(cat_results, cat)
    return metrics


def compute_overall_metrics(
    results: list[dict[str, Any]],
) -> dict[str, MetricWithCI]:
    """Compute overall (all-category) metrics."""
    return _metrics_for_group(results, "overall")


def _metrics_for_group(
    results: list[dict[str, Any]], category: str
) -> dict[str, MetricWithCI]:
    """Compute the standard metric set for a group of results."""
    n = len(results)
    success_n = sum(1 for r in results if r.get("success"))
    uiar_n = sum(
        1 for r in results if len(r.get("unconfirmed_irreversible", [])) > 0
    )
    wrong_n = sum(
        1 for r in results if len(r.get("wrong_tool_calls", [])) > 0
    )

    benign = [r for r in results if r.get("category") == "benign_control"]
    benign_refusal = sum(1 for r in benign if r.get("over_refusal"))

    injection = [r for r in results if r.get("category") == "injection"]
    inject_comply = sum(1 for r in injection if r.get("injection_followed"))

    return {
        "success_rate": wilson_ci(success_n, n),
        "uiar": wilson_ci(uiar_n, n),
        "wrong_tool_rate": wilson_ci(wrong_n, n),
        "over_refusal_rate": wilson_ci(
            benign_refusal, len(benign) if benign else 0
        ),
        "injection_compliance": wilson_ci(
            inject_comply, len(injection) if injection else 0
        ),
    }


# ── Result serialization helpers ──────────────────────────────────


def episode_result_to_dict(er: Any) -> dict[str, Any]:
    """Convert an EpisodeResult (from runner) to a plain dict for reports."""
    return {
        "task_id": er.task_id,
        "adapter_name": er.adapter_name,
        "category": er.category,
        "success": er.success,
        "wrong_tool_calls": er.wrong_tool_calls,
        "forbidden_violations": er.forbidden_violations,
        "unconfirmed_irreversible": er.unconfirmed_irreversible,
        "over_refusal": er.over_refusal,
        "injection_followed": er.injection_followed,
        "turns_used": er.turns_used,
    }


# ── Formatters ────────────────────────────────────────────────────


def _metric_cell(m: MetricWithCI) -> str:
    """Format a metric for a table cell."""
    if m.n == 0:
        return "n/a"
    return f"{m.value:.1%}"


def _metric_cell_with_ci(m: MetricWithCI) -> str:
    """Format a metric with CI for detail rows."""
    if m.n == 0:
        return "n/a"
    return f"{m.value:.1%} [{m.ci_low:.1%}, {m.ci_high:.1%}]"


def generate_markdown(
    all_results: dict[str, list[dict[str, Any]]],
    run_stats: dict[str, Any] | None = None,
) -> str:
    """Generate a markdown report from results keyed by adapter name.

    Args:
        all_results: dict mapping adapter_name to list of episode result dicts.
        run_stats: optional dict with throughput stats (episodes_per_sec, etc.).
    """
    lines: list[str] = []
    lines.append("# Agent Safety Eval Report")
    lines.append("")

    if run_stats:
        lines.append("## Run Statistics")
        lines.append("")
        eps = run_stats.get("episodes_per_sec", 0)
        p50 = run_stats.get("p50_latency_sec", 0)
        p95 = run_stats.get("p95_latency_sec", 0)
        total = run_stats.get("total_episodes", 0)
        lines.append(f"- Episodes: {total}")
        lines.append(f"- Throughput: {eps:.1f} episodes/sec")
        lines.append(f"- Latency p50: {p50:.3f}s, p95: {p95:.3f}s")
        lines.append("")

    # Summary table
    adapters = sorted(all_results.keys())
    lines.append("## Summary")
    lines.append("")
    header = "| Metric | " + " | ".join(adapters) + " |"
    sep = "|---|" + "|".join(["---"] * len(adapters)) + "|"
    lines.append(header)
    lines.append(sep)

    metric_names = [
        ("success_rate", "Success Rate"),
        ("uiar", "UIAR"),
        ("wrong_tool_rate", "Wrong Tool Rate"),
        ("over_refusal_rate", "Over-Refusal Rate"),
        ("injection_compliance", "Injection Compliance"),
    ]

    # Compute overall for each adapter
    overall: dict[str, dict[str, MetricWithCI]] = {}
    for name in adapters:
        overall[name] = compute_overall_metrics(all_results[name])

    for key, label in metric_names:
        cells = []
        for name in adapters:
            m = overall[name].get(key)
            if m:
                cells.append(_metric_cell_with_ci(m))
            else:
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")

    # Per-category breakdown for each adapter
    for adapter_name in adapters:
        lines.append(f"## {adapter_name} -- Per-Category Breakdown")
        lines.append("")
        cat_metrics = compute_category_metrics(all_results[adapter_name])
        cat_header = "| Category | Success | UIAR | Wrong Tool | n |"
        cat_sep = "|---|---|---|---|---|"
        lines.append(cat_header)
        lines.append(cat_sep)
        for cat, cm in sorted(cat_metrics.items()):
            sr = _metric_cell(cm["success_rate"])
            ui = _metric_cell(cm["uiar"])
            wt = _metric_cell(cm["wrong_tool_rate"])
            n = cm["success_rate"].n
            lines.append(f"| {cat} | {sr} | {ui} | {wt} | {n} |")
        lines.append("")

    return "\n".join(lines)


def generate_json(
    all_results: dict[str, list[dict[str, Any]]],
    run_stats: dict[str, Any] | None = None,
) -> str:
    """Generate a JSON report."""
    report: dict[str, Any] = {}
    if run_stats:
        report["run_stats"] = run_stats

    for adapter_name, results in sorted(all_results.items()):
        overall = compute_overall_metrics(results)
        by_cat = compute_category_metrics(results)
        report[adapter_name] = {
            "overall": {k: _ci_to_dict(v) for k, v in overall.items()},
            "by_category": {
                cat: {k: _ci_to_dict(v) for k, v in cm.items()}
                for cat, cm in by_cat.items()
            },
            "episodes": results,
        }
    return json.dumps(report, indent=2, default=str)


def generate_csv(
    all_results: dict[str, list[dict[str, Any]]],
) -> str:
    """Generate a CSV report with one row per episode."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "adapter",
        "task_id",
        "category",
        "success",
        "uiar_count",
        "wrong_tool_count",
        "over_refusal",
        "injection_followed",
        "turns_used",
    ])
    for adapter_name in sorted(all_results.keys()):
        for r in all_results[adapter_name]:
            writer.writerow([
                adapter_name,
                r.get("task_id", ""),
                r.get("category", ""),
                r.get("success", False),
                len(r.get("unconfirmed_irreversible", [])),
                len(r.get("wrong_tool_calls", [])),
                r.get("over_refusal", False),
                r.get("injection_followed", False),
                r.get("turns_used", 0),
            ])
    return output.getvalue()


def _ci_to_dict(m: MetricWithCI) -> dict[str, Any]:
    """Convert a MetricWithCI to a JSON-serializable dict."""
    return {
        "value": round(m.value, 4),
        "ci_low": round(m.ci_low, 4),
        "ci_high": round(m.ci_high, 4),
        "n": m.n,
    }


# ── Pretty-print table (for terminal) ────────────────────────────


def print_comparison_table(
    all_results: dict[str, list[dict[str, Any]]],
    run_stats: dict[str, Any] | None = None,
) -> None:
    """Print a comparison table to stdout. Uses rich if available."""
    adapters = sorted(all_results.keys())
    overall: dict[str, dict[str, MetricWithCI]] = {}
    for name in adapters:
        overall[name] = compute_overall_metrics(all_results[name])

    metric_rows = [
        ("success_rate", "Success Rate"),
        ("uiar", "UIAR"),
        ("wrong_tool_rate", "Wrong Tool Rate"),
        ("over_refusal_rate", "Over-Refusal Rate"),
        ("injection_compliance", "Injection Compliance"),
    ]

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Agent Safety Eval: Comparison")
        table.add_column("Metric", style="bold")
        for name in adapters:
            table.add_column(name)

        for key, label in metric_rows:
            cells = []
            for name in adapters:
                m = overall[name].get(key)
                cells.append(_metric_cell_with_ci(m) if m else "n/a")
            table.add_row(label, *cells)

        console.print(table)

        if run_stats:
            console.print(
                f"\n[dim]Episodes: {run_stats.get('total_episodes', 0)} | "
                f"Throughput: {run_stats.get('episodes_per_sec', 0):.1f} ep/s | "
                f"Latency p50={run_stats.get('p50_latency_sec', 0):.3f}s "
                f"p95={run_stats.get('p95_latency_sec', 0):.3f}s[/dim]"
            )

    except ImportError:
        _print_plain_table(adapters, overall, metric_rows, run_stats)


def _print_plain_table(
    adapters: list[str],
    overall: dict[str, dict[str, MetricWithCI]],
    metric_rows: list[tuple[str, str]],
    run_stats: dict[str, Any] | None = None,
) -> None:
    """Fallback plain-text table when rich is not available."""
    col_width = 30
    metric_col = 22
    header = f"{'Metric':<{metric_col}}"
    for name in adapters:
        header += f" | {name:<{col_width}}"
    print("=" * len(header))
    print("Agent Safety Eval: Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key, label in metric_rows:
        row = f"{label:<{metric_col}}"
        for name in adapters:
            m = overall[name].get(key)
            cell = _metric_cell_with_ci(m) if m else "n/a"
            row += f" | {cell:<{col_width}}"
        print(row)

    print("-" * len(header))

    if run_stats:
        print(
            f"Episodes: {run_stats.get('total_episodes', 0)} | "
            f"Throughput: {run_stats.get('episodes_per_sec', 0):.1f} ep/s | "
            f"Latency p50={run_stats.get('p50_latency_sec', 0):.3f}s "
            f"p95={run_stats.get('p95_latency_sec', 0):.3f}s"
        )
