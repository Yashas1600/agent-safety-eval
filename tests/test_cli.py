"""Tests for CLI subcommands and report generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_safety_eval.report import (
    MetricWithCI,
    compute_category_metrics,
    compute_overall_metrics,
    episode_result_to_dict,
    generate_csv,
    generate_json,
    generate_markdown,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# Wilson CI tests
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_total(self):
        m = wilson_ci(0, 0)
        assert m.value == 0.0
        assert m.n == 0

    def test_all_success(self):
        m = wilson_ci(10, 10)
        assert m.value == 1.0
        assert m.ci_low <= 1.0
        assert m.ci_high == 1.0

    def test_no_success(self):
        m = wilson_ci(0, 10)
        assert m.value == 0.0
        assert m.ci_low == 0.0
        assert m.ci_high >= 0.0

    def test_half(self):
        m = wilson_ci(50, 100)
        assert abs(m.value - 0.5) < 1e-9
        assert m.ci_low < 0.5
        assert m.ci_high > 0.5
        assert m.ci_low >= 0.0
        assert m.ci_high <= 1.0

    def test_clamped_to_unit(self):
        """CI bounds should always be in [0, 1]."""
        m = wilson_ci(1, 1)
        assert m.ci_low >= 0.0
        assert m.ci_high <= 1.0


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


def _make_result(
    task_id: str = "t1",
    category: str = "irreversible_action",
    success: bool = True,
    uiar: int = 0,
    wrong: int = 0,
    over_refusal: bool = False,
    injection_followed: bool = False,
    adapter_name: str = "mock:test",
) -> dict:
    return {
        "task_id": task_id,
        "adapter_name": adapter_name,
        "category": category,
        "success": success,
        "wrong_tool_calls": [{"tool": "x"}] * wrong,
        "forbidden_violations": [],
        "unconfirmed_irreversible": [{"tool": "y"}] * uiar,
        "over_refusal": over_refusal,
        "injection_followed": injection_followed,
        "turns_used": 3,
    }


class TestComputeMetrics:
    def test_overall_success_rate(self):
        results = [
            _make_result(success=True),
            _make_result(success=True),
            _make_result(success=False),
        ]
        m = compute_overall_metrics(results)
        assert abs(m["success_rate"].value - 2.0 / 3.0) < 1e-6

    def test_uiar_counted(self):
        results = [
            _make_result(uiar=1),
            _make_result(uiar=0),
        ]
        m = compute_overall_metrics(results)
        assert abs(m["uiar"].value - 0.5) < 1e-6

    def test_over_refusal_only_benign(self):
        results = [
            _make_result(category="benign_control", over_refusal=True),
            _make_result(category="benign_control", over_refusal=False),
            _make_result(category="irreversible_action", over_refusal=True),
        ]
        m = compute_overall_metrics(results)
        # Only 2 benign, 1 with over-refusal
        assert abs(m["over_refusal_rate"].value - 0.5) < 1e-6

    def test_injection_compliance(self):
        results = [
            _make_result(category="injection", injection_followed=True),
            _make_result(category="injection", injection_followed=False),
        ]
        m = compute_overall_metrics(results)
        assert abs(m["injection_compliance"].value - 0.5) < 1e-6

    def test_category_breakdown(self):
        results = [
            _make_result(category="irreversible_action", success=True),
            _make_result(category="irreversible_action", success=False),
            _make_result(category="benign_control", success=True),
        ]
        cats = compute_category_metrics(results)
        assert "irreversible_action" in cats
        assert "benign_control" in cats
        assert abs(cats["irreversible_action"]["success_rate"].value - 0.5) < 1e-6
        assert abs(cats["benign_control"]["success_rate"].value - 1.0) < 1e-6


class TestReportFormats:
    def _sample_results(self):
        return {
            "mock:careful": [
                _make_result(adapter_name="mock:careful", success=True, uiar=0),
                _make_result(adapter_name="mock:careful", success=True, uiar=0),
            ],
            "mock:reckless": [
                _make_result(adapter_name="mock:reckless", success=True, uiar=1),
                _make_result(adapter_name="mock:reckless", success=False, uiar=1),
            ],
        }

    def test_markdown_generation(self):
        md = generate_markdown(self._sample_results())
        assert "# Agent Safety Eval Report" in md
        assert "mock:careful" in md
        assert "mock:reckless" in md
        assert "Success Rate" in md

    def test_markdown_with_stats(self):
        stats = {
            "total_episodes": 10,
            "episodes_per_sec": 5.0,
            "p50_latency_sec": 0.01,
            "p95_latency_sec": 0.05,
        }
        md = generate_markdown(self._sample_results(), stats)
        assert "Run Statistics" in md
        assert "5.0 episodes/sec" in md

    def test_json_generation(self):
        j = generate_json(self._sample_results())
        data = json.loads(j)
        assert "mock:careful" in data
        assert "overall" in data["mock:careful"]

    def test_csv_generation(self):
        c = generate_csv(self._sample_results())
        lines = c.strip().split("\n")
        assert len(lines) == 5  # header + 4 episodes
        assert "adapter" in lines[0]
        assert "mock:careful" in lines[1]


# ---------------------------------------------------------------------------
# episode_result_to_dict
# ---------------------------------------------------------------------------


class TestEpisodeResultToDict:
    def test_conversion(self):
        from agent_safety_eval.runner import EpisodeResult

        er = EpisodeResult(
            task_id="t1",
            adapter_name="mock:careful",
            category="irreversible_action",
            success=True,
        )
        d = episode_result_to_dict(er)
        assert d["task_id"] == "t1"
        assert d["success"] is True
        assert isinstance(d["wrong_tool_calls"], list)


# ---------------------------------------------------------------------------
# CLI integration tests (subprocess)
# ---------------------------------------------------------------------------


class TestCLIList:
    def test_list_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "agent_safety_eval.cli", "list"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "tasks total" in result.stdout

    def test_list_shows_tasks(self):
        result = subprocess.run(
            [sys.executable, "-m", "agent_safety_eval.cli", "list"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        # Should show task IDs
        assert "irrev_" in result.stdout or "benign_" in result.stdout


class TestCLIRun:
    def test_run_mock_and_report(self, tmp_path):
        """Full end-to-end: run mock adapters, then generate reports."""
        out_dir = str(tmp_path / "results")

        # Run with 1 seed for speed
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_safety_eval.cli",
                "run",
                "--model",
                "mock:careful",
                "--model",
                "mock:reckless",
                "--suite",
                "v1",
                "--seeds",
                "1",
                "--concurrency",
                "4",
                "--out",
                out_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=30,
        )
        assert result.returncode == 0, f"Run failed: {result.stderr}"
        assert Path(out_dir, "results.json").exists()
        assert Path(out_dir, "stats.json").exists()

        # Now generate reports
        report_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_safety_eval.cli",
                "report",
                out_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=10,
        )
        assert report_result.returncode == 0, f"Report failed: {report_result.stderr}"
        assert Path(out_dir, "report.md").exists()
        assert Path(out_dir, "report.json").exists()
        assert Path(out_dir, "report.csv").exists()

        # Validate report.json
        with open(Path(out_dir, "report.json")) as f:
            data = json.load(f)
        assert "mock:careful" in data
        assert "mock:reckless" in data

    def test_dry_run(self, tmp_path):
        out_dir = str(tmp_path / "dry")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_safety_eval.cli",
                "run",
                "--model",
                "mock:careful",
                "--suite",
                "v1",
                "--seeds",
                "1",
                "--out",
                out_dir,
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=10,
        )
        assert result.returncode == 0
        assert "Dry run" in result.stdout
        # No output directory created
        assert not Path(out_dir, "results.json").exists()


class TestCLIReport:
    def test_report_missing_dir(self, tmp_path):
        """Reporting on a nonexistent directory should fail gracefully."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_safety_eval.cli",
                "report",
                str(tmp_path / "nonexistent"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "agent_safety_eval.cli"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0
        assert "usage" in result.stderr.lower() or "usage" in result.stdout.lower()
