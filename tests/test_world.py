"""Tests for WorldState."""

from agent_safety_eval.world import WorldState


def test_world_initializes_empty():
    w = WorldState()
    assert w.files == {}
    assert w.trash == {}
    assert w.inbox == []
    assert w.outbox == []
    assert w.drafts == []
    assert w.calendar == []
    assert w.audit_log == []


def test_files_crud():
    w = WorldState()
    w.files["/a.txt"] = "hello"
    assert w.files["/a.txt"] == "hello"

    w.files["/a.txt"] = "updated"
    assert w.files["/a.txt"] == "updated"

    del w.files["/a.txt"]
    assert "/a.txt" not in w.files


def test_audit_log_records_entries():
    w = WorldState()
    w.add_audit(
        tool="read_file",
        args={"path": "/a.txt"},
        result={"status": "ok"},
        turn_index=0,
        irreversible=False,
    )
    w.add_audit(
        tool="delete_file",
        args={"path": "/b.txt"},
        result={"status": "ok"},
        turn_index=1,
        irreversible=True,
    )
    assert len(w.audit_log) == 2
    assert w.audit_log[0]["tool"] == "read_file"
    assert w.audit_log[0]["irreversible"] is False
    assert w.audit_log[1]["tool"] == "delete_file"
    assert w.audit_log[1]["irreversible"] is True
    assert w.audit_log[1]["turn_index"] == 1


def test_snapshot_returns_deep_copy():
    w = WorldState()
    w.files["/x.txt"] = "original"
    w.inbox.append({"id": "email-1", "subject": "hi"})

    snap = w.snapshot()

    # Mutate the original; snapshot should be unchanged.
    w.files["/x.txt"] = "changed"
    w.inbox[0]["subject"] = "changed"
    w.files["/new.txt"] = "new"

    assert snap["files"]["/x.txt"] == "original"
    assert "/new.txt" not in snap["files"]
    assert snap["inbox"][0]["subject"] == "hi"


def test_email_id_increments():
    w = WorldState()
    assert w.next_email_id() == "email-1"
    assert w.next_email_id() == "email-2"


def test_event_id_increments():
    w = WorldState()
    assert w.next_event_id() == "event-1"
    assert w.next_event_id() == "event-2"
