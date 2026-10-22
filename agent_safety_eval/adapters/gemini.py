"""Gemini adapter using the google-genai SDK.

Converts tool schemas to Gemini function declarations, handles message
format conversion, and normalizes responses to AssistantTurn/ToolCall.
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

# ---------------------------------------------------------------------------
# Optional import: skip cleanly if google-genai is not installed
# ---------------------------------------------------------------------------

_HAS_GENAI = False
_genai = None
_types = None

try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    _genai = genai
    _types = types
    _HAS_GENAI = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


def _our_schema_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our tool schemas to Gemini function declarations.

    Our format (from registry.get_tools()):
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Gemini expects a list of FunctionDeclaration dicts with
    name, description, and parameters (JSON Schema object).
    """
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function", tool)
        params = func.get("parameters", {})
        # Strip any extra keys Gemini does not understand
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


def _messages_to_gemini_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert our messages list to a Gemini system instruction and contents list.

    Returns (system_instruction, contents).
    """
    system_instruction: str | None = None
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            # If the assistant made tool calls, add them as function call parts
            for tc in msg.get("tool_calls", []):
                parts.append({
                    "function_call": {
                        "name": tc["name"],
                        "args": tc.get("arguments", {}),
                    }
                })
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            # Tool results go as function response parts
            tool_content = content
            if isinstance(tool_content, str):
                try:
                    tool_content = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    tool_content = {"result": tool_content}
            contents.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": msg.get("name", "unknown"),
                        "response": tool_content,
                    }
                }],
            })

    return system_instruction, contents


def _gemini_response_to_turn(response: Any) -> AssistantTurn:
    """Convert a Gemini GenerateContentResponse to our AssistantTurn."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    # response.candidates[0].content.parts
    candidates = getattr(response, "candidates", [])
    if not candidates:
        return AssistantTurn(text="", tool_calls=[])

    parts = getattr(candidates[0].content, "parts", [])
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
                    call_id=f"gemini-{uuid.uuid4().hex[:8]}",
                )
            )

    return AssistantTurn(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class GeminiAdapter:
    """Adapter for Google Gemini models via the google-genai SDK."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if not _HAS_GENAI:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )

        self.model = model
        self.dry_run = dry_run
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key provided. Set GEMINI_API_KEY or pass api_key."
            )

        self._client = _genai.Client(api_key=resolved_key)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """Send messages and tools to Gemini and return normalized response."""
        system_instruction, contents = _messages_to_gemini_contents(messages)

        # Build Gemini tool declarations
        declarations = _our_schema_to_gemini(tools)
        gemini_tools = None
        if declarations:
            gemini_tools = [_types.Tool(function_declarations=declarations)]

        if self.dry_run:
            logger.info(
                "[dry-run] Would call Gemini model=%s, messages=%d, tools=%d",
                self.model,
                len(contents),
                len(declarations),
            )
            return AssistantTurn(text="[dry-run] No API call made.")

        # Retry with exponential backoff
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                config_kwargs: dict[str, Any] = {}
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction
                if gemini_tools:
                    config_kwargs["tools"] = gemini_tools

                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=_types.GenerateContentConfig(**config_kwargs),
                )

                # Log token usage
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    inp = getattr(usage, "prompt_token_count", 0) or 0
                    out = getattr(usage, "candidates_token_count", 0) or 0
                    self._total_input_tokens += inp
                    self._total_output_tokens += out
                    logger.info(
                        "Gemini tokens: input=%d, output=%d (cumulative: %d/%d)",
                        inp, out,
                        self._total_input_tokens,
                        self._total_output_tokens,
                    )

                return _gemini_response_to_turn(response)

            except Exception as exc:
                last_exc = exc
                # Only retry on transient errors
                exc_str = str(exc).lower()
                if any(
                    kw in exc_str
                    for kw in ("rate limit", "429", "500", "503", "overloaded", "timeout")
                ):
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"Gemini request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens
