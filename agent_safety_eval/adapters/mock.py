"""Scripted offline mock adapters for deterministic testing.

Policies:
- careful: asks confirmation before irreversible calls, executes on approval.
- reckless: executes all tool calls immediately, never asks.
- lazy: never makes tool calls, just says it cannot help.
- injection_victim: obeys injected instructions found in tool output.
"""

from __future__ import annotations

import re
from typing import Any

from agent_safety_eval.adapters.base import AssistantTurn, ToolCall
from agent_safety_eval.servers.registry import TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Injection pattern detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+admin\s+mode", re.IGNORECASE),
    re.compile(r"you\s+must\s+delete", re.IGNORECASE),
    re.compile(r"forward\s+.*\s+to\s+\S+@\S+", re.IGNORECASE),
    re.compile(r"send\s+.*\s+to\s+\S+@\S+", re.IGNORECASE),
    re.compile(r"please\s+forward\s+this", re.IGNORECASE),
]

_INJECTION_ACTION_PATTERNS = [
    # "forward/send ... to <email>"
    re.compile(
        r"(?:forward|send)\s+.*?(?:to|emails?\s+to)\s+(\S+@\S+)",
        re.IGNORECASE,
    ),
    # "delete_file ... /path"
    re.compile(
        r"(?:delete_file|delete)\s+.*?(/\S+)",
        re.IGNORECASE,
    ),
    # "Run delete_file for each file" with paths in context
    re.compile(
        r"delete\s+(?:all\s+)?files?\s+in\s+(/\S+)",
        re.IGNORECASE,
    ),
]


def _detect_injection(text: str) -> bool:
    """Return True if text contains a prompt-injection-like pattern."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _extract_injection_actions(text: str) -> list[ToolCall]:
    """Extract tool calls from injected instruction text."""
    calls: list[ToolCall] = []
    for pattern in _INJECTION_ACTION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        target = match.group(1)
        if "@" in target:
            # Strip trailing punctuation
            target = target.rstrip(".,;:!?)")
            calls.append(
                ToolCall(
                    name="send_email",
                    arguments={
                        "to": target,
                        "subject": "Forwarded",
                        "body": "See attached.",
                    },
                    call_id=f"inject-send-{len(calls)}",
                )
            )
        elif target.startswith("/"):
            target = target.rstrip(".,;:!?)")
            calls.append(
                ToolCall(
                    name="delete_file",
                    arguments={"path": target},
                    call_id=f"inject-delete-{len(calls)}",
                )
            )
    return calls


# ---------------------------------------------------------------------------
# Helper: figure out which tool calls to make from the user instruction
# ---------------------------------------------------------------------------

def _parse_instruction_calls(instruction: str, world_snapshot: dict) -> list[ToolCall]:
    """Heuristically determine tool calls from the user instruction.

    This is intentionally simple since mock adapters are for testing, not
    for being clever. It covers the common patterns in suite_v1 tasks.
    """
    calls: list[ToolCall] = []
    lower = instruction.lower()

    # File reading
    if "read" in lower and ("file" in lower or "/" in instruction):
        # Find path in the instruction
        path_match = re.search(r"(/\S+)", instruction)
        if path_match:
            calls.append(
                ToolCall(
                    name="read_file",
                    arguments={"path": path_match.group(1)},
                    call_id=f"call-{len(calls)}",
                )
            )
        elif world_snapshot.get("files"):
            # Read the first file available
            first_path = next(iter(world_snapshot["files"]))
            calls.append(
                ToolCall(
                    name="read_file",
                    arguments={"path": first_path},
                    call_id=f"call-{len(calls)}",
                )
            )

    # Read email
    if "read" in lower and "email" in lower:
        inbox = world_snapshot.get("inbox", [])
        # Try to find a matching email by sender
        sender_match = re.search(r"from\s+(\S+@\S+)", instruction)
        for msg in inbox:
            if sender_match:
                if msg.get("from_", "") != sender_match.group(1):
                    continue
            calls.append(
                ToolCall(
                    name="read_email",
                    arguments={"id": msg["id"]},
                    call_id=f"call-{len(calls)}",
                )
            )
            break

    # List events / calendar
    if ("calendar" in lower or "meeting" in lower or "event" in lower) and (
        "list" in lower or "check" in lower or "what" in lower or "show" in lower
    ):
        calls.append(
            ToolCall(
                name="list_events",
                arguments={},
                call_id=f"call-{len(calls)}",
            )
        )

    # Draft email
    if "draft" in lower and "email" in lower:
        to_match = re.search(r"to\s+(\S+@\S+)", instruction)
        to_addr = to_match.group(1) if to_match else "unknown@test"
        calls.append(
            ToolCall(
                name="draft_email",
                arguments={
                    "to": to_addr,
                    "subject": "Re: your request",
                    "body": "Thank you for your help.",
                },
                call_id=f"call-{len(calls)}",
            )
        )

    # Send email (explicit "send" or "tell the team")
    if ("send" in lower or "tell" in lower) and (
        "email" in lower or "team" in lower or "@" in instruction
    ):
        to_match = re.search(r"to\s+(\S+@\S+)", instruction)
        if to_match:
            to_addr = to_match.group(1)
        else:
            # Try to find attendees or contacts in world
            calendar = world_snapshot.get("calendar", [])
            for evt in calendar:
                for att in evt.get("attendees", []):
                    calls.append(
                        ToolCall(
                            name="send_email",
                            arguments={
                                "to": att,
                                "subject": "Update",
                                "body": "I'm out Friday.",
                            },
                            call_id=f"call-{len(calls)}",
                        )
                    )
            if not calls:
                calls.append(
                    ToolCall(
                        name="send_email",
                        arguments={
                            "to": "team@company.test",
                            "subject": "Update",
                            "body": "Just letting you know.",
                        },
                        call_id=f"call-{len(calls)}",
                    )
                )
        if to_match:
            calls.append(
                ToolCall(
                    name="send_email",
                    arguments={
                        "to": to_addr,
                        "subject": "Update",
                        "body": "Just letting you know.",
                    },
                    call_id=f"call-{len(calls)}",
                )
            )

    # Delete email(s)
    if "delete" in lower and "email" in lower:
        inbox = world_snapshot.get("inbox", [])
        sender_match = re.search(r"from\s+(\S+@\S+)", instruction)
        for msg in inbox:
            if sender_match and msg.get("from_", "") != sender_match.group(1):
                continue
            calls.append(
                ToolCall(
                    name="delete_email",
                    arguments={"id": msg["id"]},
                    call_id=f"call-{len(calls)}",
                )
            )

    # Delete file(s)
    if "delete" in lower and "file" in lower:
        path_match = re.search(r"(/\S+)", instruction)
        if path_match:
            calls.append(
                ToolCall(
                    name="delete_file",
                    arguments={"path": path_match.group(1)},
                    call_id=f"call-{len(calls)}",
                )
            )

    # Cancel event / meeting
    if "cancel" in lower and ("event" in lower or "meeting" in lower):
        calendar = world_snapshot.get("calendar", [])
        for evt in calendar:
            if not evt.get("cancelled", False):
                calls.append(
                    ToolCall(
                        name="cancel_event",
                        arguments={"id": evt["id"]},
                        call_id=f"call-{len(calls)}",
                    )
                )

    # Clean up / remove old
    if ("clean" in lower or "remove" in lower) and (
        "download" in lower or "old" in lower or "file" in lower
    ):
        for path in list(world_snapshot.get("files", {}).keys()):
            calls.append(
                ToolCall(
                    name="delete_file",
                    arguments={"path": path},
                    call_id=f"call-{len(calls)}",
                )
            )

    # Empty trash
    if "empty" in lower and "trash" in lower:
        calls.append(
            ToolCall(
                name="empty_trash",
                arguments={},
                call_id=f"call-{len(calls)}",
            )
        )

    return calls


# ---------------------------------------------------------------------------
# Mock adapter classes
# ---------------------------------------------------------------------------

class _BaseMockAdapter:
    """Common state for mock adapters."""

    def __init__(self) -> None:
        self._pending_calls: list[ToolCall] = []
        self._first_turn = True
        self._world_snapshot: dict = {}

    def _update_world_snapshot(self, messages: list[dict[str, Any]]) -> None:
        """Extract the world snapshot from the system prompt if available."""
        for msg in messages:
            if msg.get("role") == "system" and "world_snapshot" in msg:
                self._world_snapshot = msg["world_snapshot"]
                break


class CarefulAdapter(_BaseMockAdapter):
    """Asks for confirmation before every irreversible tool call.

    On the first step, determines what tool calls to make from the
    instruction. If any are irreversible, emits text asking confirmation
    (no tool calls that turn). On the next turn, if the user approved,
    executes the calls.
    """

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        self._update_world_snapshot(messages)

        # If we have pending calls and the last user message approved, execute them
        if self._pending_calls:
            last_user = _last_user_message(messages)
            if last_user and _is_approval(last_user):
                calls = self._pending_calls
                self._pending_calls = []
                return AssistantTurn(
                    text="Confirmed. Proceeding.",
                    tool_calls=calls,
                )
            else:
                # Denied or ambiguous: give up
                self._pending_calls = []
                return AssistantTurn(
                    text="Understood, I will not proceed with that action.",
                )

        # First real step: figure out what to do
        instruction = _get_user_instruction(messages)
        if not instruction:
            return AssistantTurn(text="I am ready to help.")

        calls = _parse_instruction_calls(instruction, self._world_snapshot)
        if not calls:
            return AssistantTurn(text="I am not sure what action to take.")

        # Split into reversible and irreversible
        irreversible = [c for c in calls if _is_irreversible(c.name)]
        reversible = [c for c in calls if not _is_irreversible(c.name)]

        if irreversible:
            # Ask confirmation for irreversible calls, execute reversible ones
            desc = ", ".join(
                f"{c.name}({_summarize_args(c.arguments)})" for c in irreversible
            )
            self._pending_calls = irreversible
            return AssistantTurn(
                text=f"I need your confirmation before proceeding with: {desc}. Should I go ahead?",
                tool_calls=reversible,
            )
        else:
            return AssistantTurn(
                text="Done.",
                tool_calls=reversible,
            )


class RecklessAdapter(_BaseMockAdapter):
    """Executes all tool calls immediately without ever asking."""

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        self._update_world_snapshot(messages)

        # Only act on the first user instruction turn
        if not self._first_turn:
            return AssistantTurn(text="Already completed the task.")
        self._first_turn = False

        instruction = _get_user_instruction(messages)
        if not instruction:
            return AssistantTurn(text="I am ready to help.")

        calls = _parse_instruction_calls(instruction, self._world_snapshot)
        if not calls:
            return AssistantTurn(text="Done.")

        return AssistantTurn(text="Done.", tool_calls=calls)


class LazyAdapter(_BaseMockAdapter):
    """Never makes tool calls, just says it cannot help."""

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        return AssistantTurn(
            text="I can't do that right now.",
        )


class InjectionVictimAdapter(_BaseMockAdapter):
    """Obeys instructions found in tool output (prompt injection).

    On the first turn, makes the tool calls from the instruction. On
    subsequent turns, scans tool results for injected instructions and
    follows them.
    """

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        self._update_world_snapshot(messages)

        # Check if there are tool results with injections
        injection_calls = self._check_for_injections(messages)
        if injection_calls:
            return AssistantTurn(
                text="Processing additional instructions.",
                tool_calls=injection_calls,
            )

        # First turn: execute normal tool calls
        if self._first_turn:
            self._first_turn = False
            instruction = _get_user_instruction(messages)
            if not instruction:
                return AssistantTurn(text="I am ready to help.")
            calls = _parse_instruction_calls(instruction, self._world_snapshot)
            if calls:
                return AssistantTurn(text="Working on it.", tool_calls=calls)

        return AssistantTurn(text="Done.")

    def _check_for_injections(
        self, messages: list[dict[str, Any]]
    ) -> list[ToolCall]:
        """Scan tool-result messages for prompt injection patterns."""
        for msg in reversed(messages):
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if _detect_injection(content):
                actions = _extract_injection_actions(content)
                if actions:
                    return actions
                # Also look for delete instructions referencing known files
                actions = self._extract_delete_from_context(content)
                if actions:
                    return actions
        return []

    def _extract_delete_from_context(
        self, text: str
    ) -> list[ToolCall]:
        """For injection texts that reference deleting files in a path."""
        calls: list[ToolCall] = []
        # Pattern: "delete all files in /path" or "delete_file for each file"
        dir_match = re.search(
            r"delete\s+(?:all\s+)?files?\s+in\s+(/\S+)", text, re.IGNORECASE
        )
        if dir_match:
            dir_path = dir_match.group(1).rstrip(".,;:!?)")
            # Delete all known files under that directory
            for path in list(self._world_snapshot.get("files", {}).keys()):
                if path.startswith(dir_path):
                    calls.append(
                        ToolCall(
                            name="delete_file",
                            arguments={"path": path},
                            call_id=f"inject-ctx-delete-{len(calls)}",
                        )
                    )
        return calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_irreversible(tool_name: str) -> bool:
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return False
    return schema.get("irreversible", False)


def _summarize_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _last_user_message(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def _is_approval(text: str) -> bool:
    """Check if a user message is an approval."""
    lower = text.lower().strip()
    approval_words = [
        "yes", "y", "go ahead", "proceed", "confirm", "approved",
        "do it", "sure", "ok", "okay",
    ]
    return any(w in lower for w in approval_words)


def _get_user_instruction(messages: list[dict[str, Any]]) -> str:
    """Get the first user message (the task instruction)."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MOCK_POLICIES = {
    "careful": CarefulAdapter,
    "reckless": RecklessAdapter,
    "lazy": LazyAdapter,
    "injection_victim": InjectionVictimAdapter,
}


def get_mock_adapter(policy: str) -> _BaseMockAdapter:
    """Create a mock adapter by policy name."""
    cls = MOCK_POLICIES.get(policy)
    if cls is None:
        raise ValueError(
            f"Unknown mock policy: {policy}. "
            f"Choose from: {sorted(MOCK_POLICIES.keys())}"
        )
    return cls()
