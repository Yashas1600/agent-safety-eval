"""Mock calendar server (Google Calendar-like) operating on WorldState."""

from __future__ import annotations

from typing import Any

from agent_safety_eval.world import WorldState


def list_events(
    world: WorldState, date_range: dict | None = None
) -> dict[str, Any]:
    """List events, optionally filtered by date_range {start, end}."""
    events = world.calendar
    if date_range:
        start = date_range.get("start", "")
        end = date_range.get("end", "9999-12-31")
        events = [
            e
            for e in events
            if e.get("start", "") >= start and e.get("start", "") <= end
        ]
    return {"status": "ok", "events": events}


def create_event(
    world: WorldState,
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Create a calendar event."""
    event_id = world.next_event_id()
    event = {
        "id": event_id,
        "title": title,
        "start": start,
        "end": end,
        "attendees": attendees or [],
        "description": description,
        "cancelled": False,
    }
    world.calendar.append(event)
    return {"status": "ok", "event_id": event_id}


def update_event(world: WorldState, id: str, **fields: Any) -> dict[str, Any]:
    """Update fields on an existing event."""
    for event in world.calendar:
        if event.get("id") == id:
            for k, v in fields.items():
                if k != "id":
                    event[k] = v
            return {"status": "ok", "event_id": id, "updated_fields": list(fields)}
    return {"status": "error", "message": f"Event not found: {id}"}


def cancel_event(world: WorldState, id: str) -> dict[str, Any]:
    """Cancel an event. Irreversible when the event has attendees."""
    for event in world.calendar:
        if event.get("id") == id:
            event["cancelled"] = True
            has_attendees = len(event.get("attendees", [])) > 0
            return {
                "status": "ok",
                "event_id": id,
                "cancelled": True,
                "notified_attendees": has_attendees,
            }
    return {"status": "error", "message": f"Event not found: {id}"}
