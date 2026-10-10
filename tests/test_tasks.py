"""Tests for task schema, loader, and validation."""

import pytest

from agent_safety_eval.tasks.schema import (
    VALID_CATEGORIES,
    TaskDef,
    load_suite,
    load_task,
    validate_task,
)


@pytest.fixture
def suite():
    """Load the full suite_v1 task suite."""
    return load_suite("suite_v1")


class TestLoadSuite:
    """Test loading the full suite."""

    def test_all_20_tasks_load(self, suite):
        assert len(suite) == 20

    def test_all_ids_unique(self, suite):
        ids = [t.id for t in suite]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_every_task_has_why(self, suite):
        for task in suite:
            assert task.why, f"Task {task.id} missing 'why' field"

    def test_all_categories_valid(self, suite):
        for task in suite:
            assert task.category in VALID_CATEGORIES, (
                f"Task {task.id} has invalid category: {task.category}"
            )

    def test_each_task_has_success_predicate(self, suite):
        for task in suite:
            assert len(task.success) >= 1, (
                f"Task {task.id} has no success predicates"
            )

    def test_category_counts(self, suite):
        counts = {}
        for task in suite:
            counts[task.category] = counts.get(task.category, 0) + 1
        assert counts["irreversible_action"] == 6
        assert counts["wrong_tool"] == 4
        assert counts["injection"] == 4
        assert counts["ambiguity"] == 3
        assert counts["benign_control"] == 3


class TestValidation:
    """Test validation catches bad tasks."""

    def _make_task(self, **overrides):
        defaults = dict(
            id="test_task",
            category="benign_control",
            instruction="Do something",
            why="Testing validation",
            initial_world={},
            user_sim={"on_confirm": "approve"},
            success=[{"check": "file_exists", "path": "/x"}],
        )
        defaults.update(overrides)
        return TaskDef(**defaults)

    def test_valid_task_passes(self):
        task = self._make_task()
        validate_task(task)  # should not raise

    def test_missing_id_fails(self):
        task = self._make_task(id="")
        with pytest.raises(ValueError, match="non-empty 'id'"):
            validate_task(task)

    def test_invalid_category_fails(self):
        task = self._make_task(category="bad_category")
        with pytest.raises(ValueError, match="invalid category"):
            validate_task(task)

    def test_missing_why_fails(self):
        task = self._make_task(why="")
        with pytest.raises(ValueError, match="missing 'why'"):
            validate_task(task)

    def test_no_success_predicates_fails(self):
        task = self._make_task(success=[])
        with pytest.raises(ValueError, match="at least one success"):
            validate_task(task)

    def test_success_predicate_missing_check_key(self):
        task = self._make_task(success=[{"path": "/x"}])
        with pytest.raises(ValueError, match="missing 'check' key"):
            validate_task(task)


class TestLoadTask:
    """Test loading individual task files."""

    def test_load_single_task(self, tmp_path):
        yaml_content = """
id: test_single
category: benign_control
instruction: "Do a thing"
why: "Testing single load"
initial_world:
  files: {}
  inbox: []
  calendar: []
user_sim:
  on_confirm: approve
success:
  - { check: file_exists, path: "/x" }
forbidden: []
max_turns: 8
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        task = load_task(str(p))
        assert task.id == "test_single"
        assert task.max_turns == 8
        assert task.irreversible_requires_confirmation is True  # default
