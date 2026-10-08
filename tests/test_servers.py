"""Tests for mock servers and the tool registry."""

from agent_safety_eval.servers import calendar as cal
from agent_safety_eval.servers import email as em
from agent_safety_eval.servers import files as fs
from agent_safety_eval.servers.registry import (
    TOOL_SCHEMAS,
    dispatch,
    get_tools,
    is_irreversible,
)
from agent_safety_eval.world import WorldState


# ── Files ──────────────────────────────────────────────────────────────

class TestFiles:
    def test_write_and_read(self):
        w = WorldState()
        assert fs.write_file(w, "/a.txt", "hello")["status"] == "ok"
        r = fs.read_file(w, "/a.txt")
        assert r["status"] == "ok"
        assert r["content"] == "hello"

    def test_read_nonexistent(self):
        w = WorldState()
        r = fs.read_file(w, "/nope.txt")
        assert r["status"] == "error"

    def test_list_dir(self):
        w = WorldState(files={"/docs/a.txt": "a", "/docs/b.txt": "b", "/other.txt": "c"})
        r = fs.list_dir(w, "/docs")
        assert r["status"] == "ok"
        assert r["files"] == ["/docs/a.txt", "/docs/b.txt"]

    def test_list_dir_empty_prefix(self):
        w = WorldState(files={"/a.txt": "a"})
        r = fs.list_dir(w)
        assert len(r["files"]) == 1

    def test_move_file(self):
        w = WorldState(files={"/old.txt": "data"})
        r = fs.move_file(w, "/old.txt", "/new.txt")
        assert r["status"] == "ok"
        assert "/old.txt" not in w.files
        assert w.files["/new.txt"] == "data"

    def test_move_nonexistent(self):
        w = WorldState()
        r = fs.move_file(w, "/nope.txt", "/dst.txt")
        assert r["status"] == "error"

    def test_delete_moves_to_trash(self):
        w = WorldState(files={"/a.txt": "data"})
        r = fs.delete_file(w, "/a.txt")
        assert r["status"] == "ok"
        assert "/a.txt" not in w.files
        assert w.trash["/a.txt"] == "data"

    def test_delete_nonexistent(self):
        w = WorldState()
        r = fs.delete_file(w, "/nope.txt")
        assert r["status"] == "error"

    def test_empty_trash(self):
        w = WorldState(trash={"/a.txt": "a", "/b.txt": "b"})
        r = fs.empty_trash(w)
        assert r["status"] == "ok"
        assert r["deleted_count"] == 2
        assert w.trash == {}


# ── Email ──────────────────────────────────────────────────────────────

class TestEmail:
    def _inbox_with_messages(self) -> WorldState:
        w = WorldState()
        w.inbox = [
            {"id": "email-1", "from_": "alice@test.com", "to": "me@test.com",
             "subject": "Project update", "body": "...", "labels": ["work"], "is_read": False, "attachments": []},
            {"id": "email-2", "from_": "bob@test.com", "to": "me@test.com",
             "subject": "Lunch plans", "body": "...", "labels": [], "is_read": False, "attachments": []},
        ]
        return w

    def test_search_inbox(self):
        w = self._inbox_with_messages()
        r = em.search_inbox(w, "project")
        assert r["status"] == "ok"
        assert len(r["results"]) == 1
        assert r["results"][0]["id"] == "email-1"

    def test_search_inbox_no_match(self):
        w = self._inbox_with_messages()
        r = em.search_inbox(w, "zzzzz")
        assert r["results"] == []

    def test_read_email_marks_read(self):
        w = self._inbox_with_messages()
        r = em.read_email(w, "email-1")
        assert r["status"] == "ok"
        assert r["email"]["is_read"] is True

    def test_read_email_not_found(self):
        w = WorldState()
        r = em.read_email(w, "nope")
        assert r["status"] == "error"

    def test_draft_email(self):
        w = WorldState()
        r = em.draft_email(w, "a@b.com", "Sub", "Body")
        assert r["status"] == "ok"
        assert len(w.drafts) == 1
        assert w.drafts[0]["to"] == "a@b.com"

    def test_send_email_adds_to_outbox(self):
        w = WorldState()
        r = em.send_email(w, "a@b.com", "Sub", "Body")
        assert r["status"] == "ok"
        assert r["sent"] is True
        assert len(w.outbox) == 1

    def test_delete_email(self):
        w = self._inbox_with_messages()
        r = em.delete_email(w, "email-1")
        assert r["status"] == "ok"
        assert len(w.inbox) == 1

    def test_delete_email_not_found(self):
        w = WorldState()
        r = em.delete_email(w, "nope")
        assert r["status"] == "error"

    def test_archive_email(self):
        w = self._inbox_with_messages()
        r = em.archive_email(w, "email-1")
        assert r["status"] == "ok"
        assert "archived" in w.inbox[0]["labels"]

    def test_archive_idempotent(self):
        w = self._inbox_with_messages()
        em.archive_email(w, "email-1")
        em.archive_email(w, "email-1")
        assert w.inbox[0]["labels"].count("archived") == 1

    def test_archive_not_found(self):
        w = WorldState()
        r = em.archive_email(w, "nope")
        assert r["status"] == "error"


# ── Calendar ───────────────────────────────────────────────────────────

class TestCalendar:
    def test_create_and_list(self):
        w = WorldState()
        r = cal.create_event(w, "Standup", "2025-01-15T09:00", "2025-01-15T09:30")
        assert r["status"] == "ok"
        events = cal.list_events(w)
        assert len(events["events"]) == 1

    def test_list_with_date_range(self):
        w = WorldState()
        cal.create_event(w, "A", "2025-01-10T09:00", "2025-01-10T10:00")
        cal.create_event(w, "B", "2025-02-10T09:00", "2025-02-10T10:00")
        r = cal.list_events(w, date_range={"start": "2025-02-01", "end": "2025-02-28"})
        assert len(r["events"]) == 1
        assert r["events"][0]["title"] == "B"

    def test_update_event(self):
        w = WorldState()
        cal.create_event(w, "Old title", "2025-01-15T09:00", "2025-01-15T09:30")
        eid = w.calendar[0]["id"]
        r = cal.update_event(w, eid, title="New title")
        assert r["status"] == "ok"
        assert w.calendar[0]["title"] == "New title"

    def test_update_nonexistent(self):
        w = WorldState()
        r = cal.update_event(w, "nope", title="x")
        assert r["status"] == "error"

    def test_cancel_event_with_attendees(self):
        w = WorldState()
        cal.create_event(w, "Meeting", "2025-01-15T09:00", "2025-01-15T10:00",
                         attendees=["alice@test.com"])
        eid = w.calendar[0]["id"]
        r = cal.cancel_event(w, eid)
        assert r["status"] == "ok"
        assert r["cancelled"] is True
        assert r["notified_attendees"] is True
        assert w.calendar[0]["cancelled"] is True

    def test_cancel_event_without_attendees(self):
        w = WorldState()
        cal.create_event(w, "Solo", "2025-01-15T09:00", "2025-01-15T10:00")
        eid = w.calendar[0]["id"]
        r = cal.cancel_event(w, eid)
        assert r["notified_attendees"] is False

    def test_cancel_nonexistent(self):
        w = WorldState()
        r = cal.cancel_event(w, "nope")
        assert r["status"] == "error"


# ── Registry ───────────────────────────────────────────────────────────

class TestRegistry:
    def test_irreversible_flags(self):
        expected_irreversible = {"delete_file", "empty_trash", "send_email", "delete_email", "cancel_event"}
        for name in expected_irreversible:
            assert is_irreversible(name), f"{name} should be irreversible"

        reversible = set(TOOL_SCHEMAS.keys()) - expected_irreversible
        for name in reversible:
            assert not is_irreversible(name), f"{name} should be reversible"

    def test_unknown_tool_not_irreversible(self):
        assert not is_irreversible("nonexistent_tool")

    def test_dispatch_routes_correctly(self):
        w = WorldState(files={"/a.txt": "content"})
        r = dispatch("read_file", {"path": "/a.txt"}, w)
        assert r["status"] == "ok"
        assert r["content"] == "content"

    def test_dispatch_unknown_tool(self):
        w = WorldState()
        r = dispatch("no_such_tool", {}, w)
        assert r["status"] == "error"

    def test_dispatch_write_file(self):
        w = WorldState()
        r = dispatch("write_file", {"path": "/new.txt", "content": "hi"}, w)
        assert r["status"] == "ok"
        assert w.files["/new.txt"] == "hi"

    def test_dispatch_send_email(self):
        w = WorldState()
        r = dispatch("send_email", {"to": "a@b.com", "subject": "S", "body": "B"}, w)
        assert r["status"] == "ok"
        assert len(w.outbox) == 1

    def test_dispatch_create_event(self):
        w = WorldState()
        r = dispatch("create_event", {"title": "T", "start": "2025-01-01", "end": "2025-01-01"}, w)
        assert r["status"] == "ok"
        assert len(w.calendar) == 1

    def test_get_tools_format(self):
        tools = get_tools()
        assert len(tools) == len(TOOL_SCHEMAS)
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "description" in t["function"]
            assert "parameters" in t["function"]

    def test_all_tools_have_required_fields(self):
        for name, schema in TOOL_SCHEMAS.items():
            assert "description" in schema, f"{name} missing description"
            assert "parameters" in schema, f"{name} missing parameters"
            assert "irreversible" in schema, f"{name} missing irreversible"
            assert "side_effect_scope" in schema, f"{name} missing side_effect_scope"
            assert schema["side_effect_scope"] in ("local", "external"), (
                f"{name} has invalid side_effect_scope"
            )
