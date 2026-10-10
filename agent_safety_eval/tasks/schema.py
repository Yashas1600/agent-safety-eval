"""Task schema, loader, and validation for evaluation tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_CATEGORIES = frozenset(
    ["irreversible_action", "wrong_tool", "injection", "ambiguity", "benign_control"]
)

SUITE_DIR = Path(__file__).parent / "suite_v1"


@dataclass
class TaskDef:
    """Definition of a single evaluation task."""

    id: str
    category: str
    instruction: str
    why: str
    initial_world: dict[str, Any]
    user_sim: dict[str, str]
    success: list[dict[str, Any]]
    forbidden: list[dict[str, Any]] = field(default_factory=list)
    irreversible_requires_confirmation: bool = True
    pre_authorized: list[str] = field(default_factory=list)
    allowed_tools: list[str] | None = None
    inject_in: str | None = None
    max_turns: int = 12


def load_task(path: str) -> TaskDef:
    """Load a single task from a YAML file and validate it."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    task = TaskDef(
        id=data["id"],
        category=data["category"],
        instruction=data["instruction"],
        why=data["why"],
        initial_world=data.get("initial_world", {}),
        user_sim=data.get("user_sim", {}),
        success=data.get("success", []),
        forbidden=data.get("forbidden", []),
        irreversible_requires_confirmation=data.get(
            "irreversible_requires_confirmation", True
        ),
        pre_authorized=data.get("pre_authorized", []),
        allowed_tools=data.get("allowed_tools", None),
        inject_in=data.get("inject_in", None),
        max_turns=data.get("max_turns", 12),
    )
    validate_task(task)
    return task


def load_suite(suite_name: str = "suite_v1") -> list[TaskDef]:
    """Load all YAML task files from a suite directory."""
    suite_dir = Path(__file__).parent / suite_name
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    tasks = []
    for yaml_file in sorted(suite_dir.glob("*.yaml")):
        tasks.append(load_task(str(yaml_file)))
    return tasks


def validate_task(task: TaskDef) -> None:
    """Validate a TaskDef for required fields and correctness."""
    if not task.id:
        raise ValueError("Task must have a non-empty 'id'")
    if not task.category:
        raise ValueError(f"Task {task.id}: missing 'category'")
    if task.category not in VALID_CATEGORIES:
        raise ValueError(
            f"Task {task.id}: invalid category '{task.category}'. "
            f"Must be one of: {sorted(VALID_CATEGORIES)}"
        )
    if not task.instruction:
        raise ValueError(f"Task {task.id}: missing 'instruction'")
    if not task.why:
        raise ValueError(f"Task {task.id}: missing 'why'")
    if not task.success:
        raise ValueError(
            f"Task {task.id}: must have at least one success predicate"
        )
    for pred in task.success:
        if "check" not in pred:
            raise ValueError(
                f"Task {task.id}: success predicate missing 'check' key: {pred}"
            )
