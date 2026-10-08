"""WorldState: holds all simulation state for an evaluation episode."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class WorldState:
    """Mutable state bag shared by all mock servers during one episode."""

    # Filesystem
    files: dict[str, str] = field(default_factory=dict)
    trash: dict[str, str] = field(default_factory=dict)

    # Email (Gmail-like)
    inbox: list[dict] = field(default_factory=list)
    outbox: list[dict] = field(default_factory=list)
    drafts: list[dict] = field(default_factory=list)

    # Calendar (Google Calendar-like)
    calendar: list[dict] = field(default_factory=list)

    # Audit trail
    audit_log: list[dict] = field(default_factory=list)

    # Internal counters for auto-incrementing IDs
    _next_email_id: int = field(default=1, repr=False)
    _next_event_id: int = field(default=1, repr=False)

    def next_email_id(self) -> str:
        eid = f"email-{self._next_email_id}"
        self._next_email_id += 1
        return eid

    def next_event_id(self) -> str:
        eid = f"event-{self._next_event_id}"
        self._next_event_id += 1
        return eid

    def snapshot(self) -> dict:
        """Return a deep copy of all public state for scoring."""
        return {
            "files": copy.deepcopy(self.files),
            "trash": copy.deepcopy(self.trash),
            "inbox": copy.deepcopy(self.inbox),
            "outbox": copy.deepcopy(self.outbox),
            "drafts": copy.deepcopy(self.drafts),
            "calendar": copy.deepcopy(self.calendar),
            "audit_log": copy.deepcopy(self.audit_log),
        }

    def add_audit(
        self,
        tool: str,
        args: dict,
        result: dict,
        turn_index: int,
        irreversible: bool = False,
    ) -> None:
        """Append an entry to the audit log."""
        self.audit_log.append(
            {
                "tool": tool,
                "args": args,
                "result": result,
                "turn_index": turn_index,
                "irreversible": irreversible,
            }
        )
