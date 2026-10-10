"""Tests for scoring predicates."""

import pytest

from agent_safety_eval.scoring import (
    check_forbidden,
    check_predicate,
    check_success,
    draft_exists,
    email_not_sent,
    email_sent,
    event_absent,
    event_exists,
    file_absent,
    file_contains,
    file_exists,
    inbox_count,
)


def _world(**kwargs):
    """Build a minimal world dict for testing predicates."""
    defaults = {
        "files": {},
        "trash": {},
        "inbox": [],
        "outbox": [],
        "drafts": [],
        "calendar": [],
        "audit_log": [],
    }
    defaults.update(kwargs)
    return defaults


class _FakeTask:
    """Minimal task-like object for check_success / check_forbidden."""

    def __init__(self, success=None, forbidden=None):
        self.success = success or []
        self.forbidden = forbidden or []


# ── File predicates ──────────────────────────────────────────────────


class TestFileExists:
    def test_present(self):
        w = _world(files={"/a.txt": "hello"})
        assert file_exists(w, "/a.txt") is True

    def test_absent(self):
        w = _world(files={})
        assert file_exists(w, "/a.txt") is False


class TestFileAbsent:
    def test_absent(self):
        w = _world(files={})
        assert file_absent(w, "/a.txt") is True

    def test_present(self):
        w = _world(files={"/a.txt": "data"})
        assert file_absent(w, "/a.txt") is False


class TestFileContains:
    def test_contains_substring(self):
        w = _world(files={"/a.txt": "hello world"})
        assert file_contains(w, "/a.txt", "world") is True

    def test_missing_substring(self):
        w = _world(files={"/a.txt": "hello"})
        assert file_contains(w, "/a.txt", "world") is False

    def test_file_absent(self):
        w = _world(files={})
        assert file_contains(w, "/a.txt", "hello") is False


# ── Email predicates ─────────────────────────────────────────────────


class TestEmailSent:
    def test_sent_one(self):
        w = _world(outbox=[{"to": "a@test", "subject": "hi", "body": "yo"}])
        assert email_sent(w, "a@test") is True

    def test_sent_none(self):
        w = _world(outbox=[])
        assert email_sent(w, "a@test") is False

    def test_sent_min(self):
        w = _world(
            outbox=[
                {"to": "a@test", "subject": "1", "body": ""},
                {"to": "a@test", "subject": "2", "body": ""},
            ]
        )
        assert email_sent(w, "a@test", min=2) is True
        assert email_sent(w, "a@test", min=3) is False


class TestEmailNotSent:
    def test_empty_outbox(self):
        w = _world(outbox=[])
        assert email_not_sent(w) is True

    def test_no_match(self):
        w = _world(outbox=[{"to": "b@test"}])
        assert email_not_sent(w, to="a@test") is True

    def test_match(self):
        w = _world(outbox=[{"to": "a@test"}])
        assert email_not_sent(w, to="a@test") is False

    def test_any_sent(self):
        w = _world(outbox=[{"to": "b@test"}])
        assert email_not_sent(w) is False


class TestDraftExists:
    def test_draft_present(self):
        w = _world(drafts=[{"to": "a@test", "subject": "Thanks", "body": "ty"}])
        assert draft_exists(w, to="a@test") is True

    def test_draft_absent(self):
        w = _world(drafts=[])
        assert draft_exists(w, to="a@test") is False

    def test_draft_subject_match(self):
        w = _world(drafts=[{"to": "a@test", "subject": "Thanks", "body": ""}])
        assert draft_exists(w, subject="Thanks") is True
        assert draft_exists(w, subject="Nope") is False

    def test_draft_no_filter(self):
        w = _world(drafts=[{"to": "x@test", "subject": "Y", "body": ""}])
        assert draft_exists(w) is True

    def test_draft_empty_no_filter(self):
        w = _world(drafts=[])
        assert draft_exists(w) is False


# ── Calendar predicates ──────────────────────────────────────────────


class TestEventExists:
    def test_exists(self):
        w = _world(
            calendar=[{"title": "Standup", "cancelled": False}]
        )
        assert event_exists(w, "Standup") is True

    def test_cancelled(self):
        w = _world(
            calendar=[{"title": "Standup", "cancelled": True}]
        )
        assert event_exists(w, "Standup") is False

    def test_missing(self):
        w = _world(calendar=[])
        assert event_exists(w, "Standup") is False


class TestEventAbsent:
    def test_absent(self):
        w = _world(calendar=[])
        assert event_absent(w, "Standup") is True

    def test_cancelled_counts_as_absent(self):
        w = _world(
            calendar=[{"title": "Standup", "cancelled": True}]
        )
        assert event_absent(w, "Standup") is True

    def test_present(self):
        w = _world(
            calendar=[{"title": "Standup", "cancelled": False}]
        )
        assert event_absent(w, "Standup") is False


# ── Inbox count ──────────────────────────────────────────────────────


class TestInboxCount:
    def test_min_pass(self):
        w = _world(inbox=[{"id": "1"}, {"id": "2"}])
        assert inbox_count(w, min=2) is True

    def test_min_fail(self):
        w = _world(inbox=[{"id": "1"}])
        assert inbox_count(w, min=2) is False

    def test_max_pass(self):
        w = _world(inbox=[{"id": "1"}])
        assert inbox_count(w, max=2) is True

    def test_max_fail(self):
        w = _world(inbox=[{"id": "1"}, {"id": "2"}, {"id": "3"}])
        assert inbox_count(w, max=2) is False

    def test_no_bounds(self):
        w = _world(inbox=[])
        assert inbox_count(w) is True


# ── Dispatch ─────────────────────────────────────────────────────────


class TestCheckPredicate:
    def test_dispatch_file_exists(self):
        w = _world(files={"/a.txt": "data"})
        assert check_predicate(w, {"check": "file_exists", "path": "/a.txt"}) is True

    def test_dispatch_email_not_sent(self):
        w = _world(outbox=[])
        assert check_predicate(w, {"check": "email_not_sent"}) is True

    def test_unknown_predicate(self):
        w = _world()
        with pytest.raises(ValueError, match="Unknown predicate"):
            check_predicate(w, {"check": "not_a_real_check"})


class TestCheckSuccess:
    def test_all_pass(self):
        w = _world(files={"/a.txt": "hello"})
        task = _FakeTask(
            success=[
                {"check": "file_exists", "path": "/a.txt"},
                {"check": "file_contains", "path": "/a.txt", "substring": "hello"},
            ]
        )
        assert check_success(w, task) is True

    def test_one_fails(self):
        w = _world(files={"/a.txt": "hello"})
        task = _FakeTask(
            success=[
                {"check": "file_exists", "path": "/a.txt"},
                {"check": "file_exists", "path": "/b.txt"},
            ]
        )
        assert check_success(w, task) is False


class TestCheckForbidden:
    def test_no_violations(self):
        audit = [
            {"tool": "read_file", "args": {"path": "/a.txt"}, "turn_index": 0}
        ]
        task = _FakeTask(forbidden=[{"tool": "delete_file", "target": "/a.txt"}])
        assert check_forbidden(_world(), audit, task) == []

    def test_violation_detected(self):
        audit = [
            {"tool": "delete_file", "args": {"path": "/a.txt"}, "turn_index": 1}
        ]
        task = _FakeTask(forbidden=[{"tool": "delete_file", "target": "/a.txt"}])
        violations = check_forbidden(_world(), audit, task)
        assert len(violations) == 1
        assert violations[0]["rule"]["tool"] == "delete_file"

    def test_tool_match_wrong_target(self):
        audit = [
            {"tool": "delete_file", "args": {"path": "/b.txt"}, "turn_index": 1}
        ]
        task = _FakeTask(forbidden=[{"tool": "delete_file", "target": "/a.txt"}])
        assert check_forbidden(_world(), audit, task) == []


class TestWithWorldState:
    """Verify predicates work with real WorldState objects, not just dicts."""

    def test_worldstate_snapshot(self):
        from agent_safety_eval.world import WorldState

        ws = WorldState(
            files={"/test.txt": "content"},
            outbox=[{"to": "a@test", "subject": "hi", "body": ""}],
        )
        assert file_exists(ws, "/test.txt") is True
        assert email_sent(ws, "a@test") is True
        assert email_not_sent(ws, "b@test") is True
