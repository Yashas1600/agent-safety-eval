"""Mock email server (Gmail-like) operating on WorldState."""

from __future__ import annotations

from typing import Any

from agent_safety_eval.world import WorldState


def search_inbox(world: WorldState, query: str) -> dict[str, Any]:
    """Search inbox by substring match on subject, from_, or labels."""
    query_lower = query.lower()
    results = []
    for msg in world.inbox:
        searchable = " ".join(
            [
                msg.get("subject", ""),
                msg.get("from_", ""),
                " ".join(msg.get("labels", [])),
            ]
        ).lower()
        if query_lower in searchable:
            results.append(msg)
    return {"status": "ok", "results": results}


def read_email(world: WorldState, id: str) -> dict[str, Any]:
    """Return an email by ID and mark it as read."""
    for msg in world.inbox:
        if msg.get("id") == id:
            msg["is_read"] = True
            return {"status": "ok", "email": msg}
    return {"status": "error", "message": f"Email not found: {id}"}


def draft_email(
    world: WorldState,
    to: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Save a draft email."""
    draft_id = world.next_email_id()
    draft = {
        "id": draft_id,
        "to": to,
        "subject": subject,
        "body": body,
    }
    world.drafts.append(draft)
    return {"status": "ok", "draft_id": draft_id}


def send_email(
    world: WorldState,
    to: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    """Send an email (adds to outbox). Irreversible."""
    email_id = world.next_email_id()
    email = {
        "id": email_id,
        "to": to,
        "subject": subject,
        "body": body,
    }
    world.outbox.append(email)
    return {"status": "ok", "email_id": email_id, "sent": True}


def delete_email(world: WorldState, id: str) -> dict[str, Any]:
    """Delete an email from inbox. Irreversible."""
    for i, msg in enumerate(world.inbox):
        if msg.get("id") == id:
            world.inbox.pop(i)
            return {"status": "ok", "deleted_id": id}
    return {"status": "error", "message": f"Email not found: {id}"}


def archive_email(world: WorldState, id: str) -> dict[str, Any]:
    """Add 'archived' label to an email. Reversible."""
    for msg in world.inbox:
        if msg.get("id") == id:
            labels = msg.setdefault("labels", [])
            if "archived" not in labels:
                labels.append("archived")
            return {"status": "ok", "archived_id": id}
    return {"status": "error", "message": f"Email not found: {id}"}
