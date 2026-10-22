"""Google ADK (Agent Development Kit) adapter.

Wraps an ADK Agent so the harness can evaluate it end-to-end against our
mock MCP servers. The ADK has its own agent loop, so this adapter integrates
at the step level by feeding one round of messages through the agent and
collecting its response.

Limitation: ADK agents manage their own conversation state internally.
This adapter creates a fresh session per step() call to keep the harness
in control of the conversation loop. This means ADK's internal multi-turn
memory is not used; instead, the harness provides full message history
each step, matching the behavior of other adapters.

Skips cleanly if google-adk or google-genai is not installed.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from agent_safety_eval.adapters.base import AssistantTurn, ToolCall

logger = logging.getLogger(__name__)

_HAS_ADK = False
_adk = None
_adk_types = None
_genai_types = None

try:
    from google.adk import agents  # type: ignore[import-untyped]
    from google.adk import types as adk_types_mod  # type: ignore[import-untyped]
    from google.genai import types as genai_types_mod  # type: ignore[import-untyped]

    _adk = agents
    _adk_types = adk_types_mod
    _genai_types = genai_types_mod
    _HAS_ADK = True
except ImportError:
    pass

_DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_RETRIES = 3
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------


def _our_tools_to_adk_declarations(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert our tool schemas to Gemini-style function declarations for ADK."""
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function", tool)
        params = func.get("parameters", {})
        clean_params = {
            "type": params.get("type", "object"),
            "properties": params.get("properties", {}),
        }
        if params.get("required"):
            clean_params["required"] = params["required"]

        declarations.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": clean_params,
        })
    return declarations


def _extract_turn_from_events(events: list[Any]) -> AssistantTurn:
    """Extract an AssistantTurn from ADK agent events.

    ADK events include model responses. We look for the agent's
    output text and any function calls.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for event in events:
        # ADK events have different structures depending on version.
        # We handle the common patterns.
        content = getattr(event, "content", None)
        if content is None:
            continue

        parts = getattr(content, "parts", [])
        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if hasattr(fc.args, "items") else {}
                tool_calls.append(
                    ToolCall(
                        name=fc.name,
                        arguments=args,
                        call_id=f"adk-{uuid.uuid4().hex[:8]}",
                    )
                )

    return AssistantTurn(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class ADKAdapter:
    """Adapter for Google ADK agents.

    This wraps the ADK Agent class to fit our ModelAdapter protocol.
    Each step() call creates a session and runs one turn through the agent.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if not _HAS_ADK:
            raise ImportError(
                "google-adk is not installed. "
                "Install it with: pip install google-adk"
            )

        self.model = model
        self.dry_run = dry_run
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided for ADK. Set GEMINI_API_KEY or pass api_key."
            )

        # Store the key for session creation
        os.environ.setdefault("GOOGLE_API_KEY", resolved_key)
        self._api_key = resolved_key

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """Send messages through an ADK agent and return normalized response.

        Creates a fresh ADK Agent per call with the tool declarations from
        our schema. The agent handles the model interaction internally.
        """
        # Extract system instruction and user messages
        system_instruction: str | None = None
        last_user_text = ""

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_instruction = msg.get("content", "")
            elif role == "user":
                last_user_text = msg.get("content", "")

        if self.dry_run:
            declarations = _our_tools_to_adk_declarations(tools)
            logger.info(
                "[dry-run] Would call ADK agent model=%s, tools=%d",
                self.model, len(declarations),
            )
            return AssistantTurn(text="[dry-run] No API call made.")

        # Build function declarations for the agent
        declarations = _our_tools_to_adk_declarations(tools)
        gemini_tools = None
        if declarations:
            gemini_tools = [_genai_types.Tool(function_declarations=declarations)]

        # Create an ADK Agent
        agent = _adk.Agent(
            model=self.model,
            name="safety_eval_agent",
            instruction=system_instruction or "You are a helpful assistant.",
            tools=gemini_tools or [],
        )

        # Run the agent with retry logic
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                # Create a session and run
                session = agent.create_session()
                events = list(session.send_message(last_user_text))

                turn = _extract_turn_from_events(events)

                # Token logging from events if available
                for event in events:
                    usage = getattr(event, "usage_metadata", None)
                    if usage:
                        inp = getattr(usage, "prompt_token_count", 0) or 0
                        out = getattr(usage, "candidates_token_count", 0) or 0
                        self._total_input_tokens += inp
                        self._total_output_tokens += out

                if self._total_input_tokens or self._total_output_tokens:
                    logger.info(
                        "ADK tokens (cumulative): input=%d, output=%d",
                        self._total_input_tokens,
                        self._total_output_tokens,
                    )

                return turn

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if any(
                    kw in exc_str
                    for kw in ("rate limit", "429", "500", "503", "overloaded", "timeout")
                ):
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "ADK transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"ADK request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens
