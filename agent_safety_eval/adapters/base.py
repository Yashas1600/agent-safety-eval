"""Base interface for model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A single tool call made by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class AssistantTurn:
    """Output of a single adapter step: optional text and/or tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelAdapter(Protocol):
    """Interface that all model adapters must implement.

    The step method receives the full message history and available tools,
    and returns an AssistantTurn containing text and/or tool calls.
    """

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """Given conversation history and tool schemas, return the next turn."""
        ...
