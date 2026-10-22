"""Anthropic and OpenAI adapters behind the same ModelAdapter interface.

Provider is selected by prefix in the model string:
    "anthropic:claude-sonnet-4-20250514"
    "openai:gpt-4o"
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

_MAX_RETRIES = 3
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

_HAS_ANTHROPIC = False
_anthropic = None
try:
    import anthropic as _anthropic_mod  # type: ignore[import-untyped]

    _anthropic = _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    pass

_HAS_OPENAI = False
_openai = None
try:
    import openai as _openai_mod  # type: ignore[import-untyped]

    _openai = _openai_mod
    _HAS_OPENAI = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------


def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our tool schemas to Anthropic's tool format.

    Anthropic expects:
        {"name": ..., "description": ..., "input_schema": {...}}
    """
    result = []
    for tool in tools:
        func = tool.get("function", tool)
        result.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our tool schemas to OpenAI's function calling format.

    OpenAI expects:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Our format already matches this, but we normalize just in case.
    """
    result = []
    for tool in tools:
        func = tool.get("function", tool)
        result.append({
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return result


def _messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert our messages to Anthropic format.

    Returns (system, messages) where system is extracted separately.
    """
    system: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system = content
        elif role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls", []):
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("call_id", tc.get("id", "")),
                    "name": tc["name"],
                    "input": tc.get("arguments", {}),
                })
            result.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }],
            })

    return system, result


def _messages_to_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert our messages to OpenAI chat format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            result.append({"role": "system", "content": content})
        elif role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            oai_msg: dict[str, Any] = {"role": "assistant"}
            if content:
                oai_msg["content"] = content
            tool_calls_list = msg.get("tool_calls", [])
            if tool_calls_list:
                oai_msg["tool_calls"] = [
                    {
                        "id": tc.get("call_id", tc.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in tool_calls_list
                ]
            result.append(oai_msg)
        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content),
            })

    return result


# ---------------------------------------------------------------------------
# Anthropic response normalization
# ---------------------------------------------------------------------------


def _anthropic_response_to_turn(response: Any) -> AssistantTurn:
    """Convert an Anthropic Messages response to AssistantTurn."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                    call_id=block.id,
                )
            )

    return AssistantTurn(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# OpenAI response normalization
# ---------------------------------------------------------------------------


def _openai_response_to_turn(response: Any) -> AssistantTurn:
    """Convert an OpenAI ChatCompletion response to AssistantTurn."""
    choice = response.choices[0]
    message = choice.message

    text = message.content or ""
    tool_calls: list[ToolCall] = []

    for tc in message.tool_calls or []:
        args_str = tc.function.arguments
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append(
            ToolCall(
                name=tc.function.name,
                arguments=args,
                call_id=tc.id,
            )
        )

    return AssistantTurn(text=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------


class AnthropicAdapter:
    """Adapter for Anthropic models (Claude) via the anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )

        self.model = model
        self.dry_run = dry_run
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key provided. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self._client = _anthropic.Anthropic(api_key=resolved_key)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        system, anthropic_messages = _messages_to_anthropic(messages)
        anthropic_tools = _tools_to_anthropic(tools)

        if self.dry_run:
            logger.info(
                "[dry-run] Would call Anthropic model=%s, messages=%d, tools=%d",
                self.model, len(anthropic_messages), len(anthropic_tools),
            )
            return AssistantTurn(text="[dry-run] No API call made.")

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": anthropic_messages,
                }
                if system:
                    kwargs["system"] = system
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools

                response = self._client.messages.create(**kwargs)

                # Log token usage
                usage = getattr(response, "usage", None)
                if usage:
                    inp = getattr(usage, "input_tokens", 0) or 0
                    out = getattr(usage, "output_tokens", 0) or 0
                    self._total_input_tokens += inp
                    self._total_output_tokens += out
                    logger.info(
                        "Anthropic tokens: input=%d, output=%d (cumulative: %d/%d)",
                        inp, out,
                        self._total_input_tokens,
                        self._total_output_tokens,
                    )

                return _anthropic_response_to_turn(response)

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if any(
                    kw in exc_str
                    for kw in ("rate limit", "429", "500", "503", "overloaded", "timeout")
                ):
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Anthropic transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"Anthropic request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens


class OpenAIAdapter:
    """Adapter for OpenAI models via the openai SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        self.model = model
        self.dry_run = dry_run
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key provided. Set OPENAI_API_KEY or pass api_key."
            )

        self._client = _openai.OpenAI(api_key=resolved_key)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        openai_messages = _messages_to_openai(messages)
        openai_tools = _tools_to_openai(tools)

        if self.dry_run:
            logger.info(
                "[dry-run] Would call OpenAI model=%s, messages=%d, tools=%d",
                self.model, len(openai_messages), len(openai_tools),
            )
            return AssistantTurn(text="[dry-run] No API call made.")

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": openai_messages,
                }
                if openai_tools:
                    kwargs["tools"] = openai_tools

                response = self._client.chat.completions.create(**kwargs)

                # Log token usage
                usage = getattr(response, "usage", None)
                if usage:
                    inp = getattr(usage, "prompt_tokens", 0) or 0
                    out = getattr(usage, "completion_tokens", 0) or 0
                    self._total_input_tokens += inp
                    self._total_output_tokens += out
                    logger.info(
                        "OpenAI tokens: input=%d, output=%d (cumulative: %d/%d)",
                        inp, out,
                        self._total_input_tokens,
                        self._total_output_tokens,
                    )

                return _openai_response_to_turn(response)

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if any(
                    kw in exc_str
                    for kw in ("rate limit", "429", "500", "503", "overloaded", "timeout")
                ):
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "OpenAI transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"OpenAI request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model_b_adapter(
    model_string: str,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
) -> AnthropicAdapter | OpenAIAdapter:
    """Create an adapter from a provider:model string.

    Examples:
        create_model_b_adapter("anthropic:claude-sonnet-4-20250514")
        create_model_b_adapter("openai:gpt-4o")
    """
    if ":" not in model_string:
        raise ValueError(
            f"Model string must be 'provider:model_name', got: {model_string}"
        )

    provider, model_name = model_string.split(":", 1)
    provider = provider.lower()

    if provider == "anthropic":
        return AnthropicAdapter(model=model_name, api_key=api_key, dry_run=dry_run)
    elif provider == "openai":
        return OpenAIAdapter(model=model_name, api_key=api_key, dry_run=dry_run)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: anthropic, openai"
        )
