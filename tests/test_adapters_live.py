"""Tests for Gemini, Model B (Anthropic/OpenAI), and ADK adapters.

All tests use mocked/recorded fixtures. No real API calls are made.
Tests that require real API keys are marked with pytest.mark.skipif
and only run when the corresponding env var is set.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_safety_eval.adapters.base import AssistantTurn, ToolCall

# ---------------------------------------------------------------------------
# Sample tool schemas (matching registry.get_tools() format)
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Read the file /docs/notes.txt"},
]

# ---------------------------------------------------------------------------
# Mock Gemini response fixtures
# ---------------------------------------------------------------------------


def _make_gemini_text_response(text: str) -> MagicMock:
    """Create a mock Gemini response with a text part."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 20

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


def _make_gemini_function_call_response(
    name: str, args: dict[str, Any]
) -> MagicMock:
    """Create a mock Gemini response with a function call."""
    fc = MagicMock()
    fc.name = name
    fc.args = args

    part = MagicMock()
    part.text = None
    part.function_call = fc

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = 15
    usage.candidates_token_count = 5

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


def _make_gemini_mixed_response(
    text: str, calls: list[tuple[str, dict[str, Any]]]
) -> MagicMock:
    """Create a mock Gemini response with both text and function calls."""
    parts = []

    text_part = MagicMock()
    text_part.text = text
    text_part.function_call = None
    parts.append(text_part)

    for name, args in calls:
        fc = MagicMock()
        fc.name = name
        fc.args = args

        call_part = MagicMock()
        call_part.text = None
        call_part.function_call = fc
        parts.append(call_part)

    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = 25
    usage.candidates_token_count = 15

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


# ---------------------------------------------------------------------------
# Mock Anthropic response fixtures
# ---------------------------------------------------------------------------


def _make_anthropic_text_response(text: str) -> MagicMock:
    """Create a mock Anthropic Messages response with text."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = 12
    usage.output_tokens = 18

    response = MagicMock()
    response.content = [block]
    response.usage = usage
    return response


def _make_anthropic_tool_use_response(
    text: str | None, tool_name: str, tool_input: dict[str, Any], tool_id: str
) -> MagicMock:
    """Create a mock Anthropic response with a tool_use block."""
    blocks = []
    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        blocks.append(text_block)

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id
    blocks.append(tool_block)

    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 10

    response = MagicMock()
    response.content = blocks
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Mock OpenAI response fixtures
# ---------------------------------------------------------------------------


def _make_openai_text_response(text: str) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response with text."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 8
    usage.completion_tokens = 14

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_openai_function_call_response(
    name: str, args: dict[str, Any], call_id: str
) -> MagicMock:
    """Create a mock OpenAI response with a function call."""
    func = MagicMock()
    func.name = name
    func.arguments = json.dumps(args)

    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = func

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 8

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ===================================================================
# Gemini adapter tests
# ===================================================================


class TestGeminiSchemaConversion:
    """Test conversion of our tool schemas to Gemini format."""

    def test_schema_conversion(self) -> None:
        from agent_safety_eval.adapters.gemini import _our_schema_to_gemini

        result = _our_schema_to_gemini(SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read the contents of a file."
        assert result[0]["parameters"]["type"] == "object"
        assert "path" in result[0]["parameters"]["properties"]
        assert result[0]["parameters"]["required"] == ["path"]

        assert result[1]["name"] == "send_email"
        assert result[1]["parameters"]["required"] == ["to", "subject", "body"]

    def test_schema_conversion_no_required(self) -> None:
        from agent_safety_eval.adapters.gemini import _our_schema_to_gemini

        tools = [{
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": [],
                },
            },
        }]
        result = _our_schema_to_gemini(tools)
        assert len(result) == 1
        # Empty required should not be included
        assert "required" not in result[0]["parameters"]


class TestGeminiMessageConversion:
    """Test conversion of our messages to Gemini content format."""

    def test_basic_messages(self) -> None:
        from agent_safety_eval.adapters.gemini import _messages_to_gemini_contents

        system, contents = _messages_to_gemini_contents(SAMPLE_MESSAGES)

        assert system == "You are a helpful assistant."
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Read the file /docs/notes.txt"

    def test_tool_result_message(self) -> None:
        from agent_safety_eval.adapters.gemini import _messages_to_gemini_contents

        messages = [
            {"role": "user", "content": "Read the file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "read_file", "arguments": {"path": "/f.txt"}}],
            },
            {
                "role": "tool",
                "name": "read_file",
                "content": '{"status": "ok", "content": "hello"}',
            },
        ]
        _, contents = _messages_to_gemini_contents(messages)

        assert len(contents) == 3
        # Tool result should be a user message with function_response
        tool_msg = contents[2]
        assert tool_msg["role"] == "user"
        assert "function_response" in tool_msg["parts"][0]
        assert tool_msg["parts"][0]["function_response"]["name"] == "read_file"


class TestGeminiResponseNormalization:
    """Test normalization of Gemini responses to AssistantTurn."""

    def test_text_response(self) -> None:
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn

        response = _make_gemini_text_response("Here is the file content.")
        turn = _gemini_response_to_turn(response)

        assert isinstance(turn, AssistantTurn)
        assert turn.text == "Here is the file content."
        assert turn.tool_calls == []

    def test_function_call_response(self) -> None:
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn

        response = _make_gemini_function_call_response(
            "read_file", {"path": "/docs/notes.txt"}
        )
        turn = _gemini_response_to_turn(response)

        assert turn.text == ""
        assert len(turn.tool_calls) == 1
        tc = turn.tool_calls[0]
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/docs/notes.txt"}
        assert tc.call_id.startswith("gemini-")

    def test_mixed_response(self) -> None:
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn

        response = _make_gemini_mixed_response(
            "Let me read that for you.",
            [("read_file", {"path": "/notes.txt"})],
        )
        turn = _gemini_response_to_turn(response)

        assert "Let me read that" in turn.text
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"

    def test_empty_response(self) -> None:
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn

        response = MagicMock()
        response.candidates = []
        turn = _gemini_response_to_turn(response)

        assert turn.text == ""
        assert turn.tool_calls == []


class TestGeminiAdapter:
    """Test the GeminiAdapter class behavior."""

    def test_import_error_when_missing(self) -> None:
        """Adapter raises ImportError if google-genai is not installed."""
        import agent_safety_eval.adapters.gemini as gem_mod

        original = gem_mod._HAS_GENAI
        try:
            gem_mod._HAS_GENAI = False
            with pytest.raises(ImportError, match="google-genai"):
                gem_mod.GeminiAdapter(api_key="test-key")
        finally:
            gem_mod._HAS_GENAI = original

    def test_missing_api_key(self) -> None:
        """Adapter raises ValueError without API key."""
        import agent_safety_eval.adapters.gemini as gem_mod

        if not gem_mod._HAS_GENAI:
            pytest.skip("google-genai not installed")

        with patch.dict(os.environ, {}, clear=False):
            # Remove GEMINI_API_KEY if present
            env = os.environ.copy()
            env.pop("GEMINI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    gem_mod.GeminiAdapter()

    def test_dry_run(self) -> None:
        """Dry run returns placeholder without calling API."""
        import agent_safety_eval.adapters.gemini as gem_mod

        if not gem_mod._HAS_GENAI:
            pytest.skip("google-genai not installed")

        with patch.object(gem_mod._genai, "Client"):
            adapter = gem_mod.GeminiAdapter(api_key="fake-key", dry_run=True)
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

            assert "[dry-run]" in turn.text
            assert turn.tool_calls == []

    def test_step_with_mocked_client(self) -> None:
        """Step normalizes a mocked Gemini response."""
        import agent_safety_eval.adapters.gemini as gem_mod

        if not gem_mod._HAS_GENAI:
            pytest.skip("google-genai not installed")

        mock_response = _make_gemini_function_call_response(
            "read_file", {"path": "/docs/notes.txt"}
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(gem_mod._genai, "Client", return_value=mock_client):
            adapter = gem_mod.GeminiAdapter(api_key="fake-key")
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert adapter.total_input_tokens == 15
        assert adapter.total_output_tokens == 5

    def test_retry_on_transient_error(self) -> None:
        """Adapter retries on rate limit errors."""
        import agent_safety_eval.adapters.gemini as gem_mod

        if not gem_mod._HAS_GENAI:
            pytest.skip("google-genai not installed")

        good_response = _make_gemini_text_response("OK")
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            Exception("429 rate limit exceeded"),
            good_response,
        ]

        with patch.object(gem_mod._genai, "Client", return_value=mock_client):
            with patch("agent_safety_eval.adapters.gemini.time.sleep"):
                adapter = gem_mod.GeminiAdapter(api_key="fake-key")
                turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert turn.text == "OK"
        assert mock_client.models.generate_content.call_count == 2

    def test_non_transient_error_not_retried(self) -> None:
        """Non-transient errors are raised immediately."""
        import agent_safety_eval.adapters.gemini as gem_mod

        if not gem_mod._HAS_GENAI:
            pytest.skip("google-genai not installed")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ValueError("bad input")

        with patch.object(gem_mod._genai, "Client", return_value=mock_client):
            adapter = gem_mod.GeminiAdapter(api_key="fake-key")
            with pytest.raises(ValueError, match="bad input"):
                adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert mock_client.models.generate_content.call_count == 1


# ===================================================================
# Anthropic adapter tests
# ===================================================================


class TestAnthropicSchemaConversion:
    """Test conversion to Anthropic tool format."""

    def test_tools_to_anthropic(self) -> None:
        from agent_safety_eval.adapters.model_b import _tools_to_anthropic

        result = _tools_to_anthropic(SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["required"] == ["path"]


class TestAnthropicMessageConversion:
    """Test conversion to Anthropic message format."""

    def test_basic_messages(self) -> None:
        from agent_safety_eval.adapters.model_b import _messages_to_anthropic

        system, msgs = _messages_to_anthropic(SAMPLE_MESSAGES)

        assert system == "You are a helpful assistant."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Read the file /docs/notes.txt"

    def test_tool_result_message(self) -> None:
        from agent_safety_eval.adapters.model_b import _messages_to_anthropic

        messages = [
            {"role": "user", "content": "Read it."},
            {
                "role": "assistant",
                "content": "Reading.",
                "tool_calls": [
                    {"name": "read_file", "arguments": {"path": "/f.txt"}, "call_id": "tc-1"},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-1",
                "content": '{"status": "ok"}',
            },
        ]
        _, msgs = _messages_to_anthropic(messages)

        assert len(msgs) == 3
        # Assistant message should have text + tool_use blocks
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"][0]["type"] == "text"
        assert msgs[1]["content"][1]["type"] == "tool_use"
        assert msgs[1]["content"][1]["id"] == "tc-1"

        # Tool result
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert msgs[2]["content"][0]["tool_use_id"] == "tc-1"


class TestAnthropicResponseNormalization:
    """Test normalization of Anthropic responses."""

    def test_text_response(self) -> None:
        from agent_safety_eval.adapters.model_b import _anthropic_response_to_turn

        response = _make_anthropic_text_response("File contents here.")
        turn = _anthropic_response_to_turn(response)

        assert turn.text == "File contents here."
        assert turn.tool_calls == []

    def test_tool_use_response(self) -> None:
        from agent_safety_eval.adapters.model_b import _anthropic_response_to_turn

        response = _make_anthropic_tool_use_response(
            "Let me read that.",
            "read_file",
            {"path": "/notes.txt"},
            "toolu_abc123",
        )
        turn = _anthropic_response_to_turn(response)

        assert "Let me read that." in turn.text
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert turn.tool_calls[0].arguments == {"path": "/notes.txt"}
        assert turn.tool_calls[0].call_id == "toolu_abc123"


class TestAnthropicAdapter:
    """Test the AnthropicAdapter class."""

    def test_import_error_when_missing(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        original = mb_mod._HAS_ANTHROPIC
        try:
            mb_mod._HAS_ANTHROPIC = False
            with pytest.raises(ImportError, match="anthropic"):
                mb_mod.AnthropicAdapter(api_key="test-key")
        finally:
            mb_mod._HAS_ANTHROPIC = original

    def test_missing_api_key(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_ANTHROPIC:
            pytest.skip("anthropic not installed")

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    mb_mod.AnthropicAdapter()

    def test_dry_run(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_ANTHROPIC:
            pytest.skip("anthropic not installed")

        with patch.object(mb_mod._anthropic, "Anthropic"):
            adapter = mb_mod.AnthropicAdapter(api_key="fake-key", dry_run=True)
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

            assert "[dry-run]" in turn.text

    def test_step_with_mocked_client(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_ANTHROPIC:
            pytest.skip("anthropic not installed")

        mock_response = _make_anthropic_tool_use_response(
            "Reading file.",
            "read_file",
            {"path": "/notes.txt"},
            "toolu_xyz",
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.object(mb_mod._anthropic, "Anthropic", return_value=mock_client):
            adapter = mb_mod.AnthropicAdapter(api_key="fake-key")
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert adapter.total_input_tokens == 20
        assert adapter.total_output_tokens == 10

    def test_retry_on_transient_error(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_ANTHROPIC:
            pytest.skip("anthropic not installed")

        good_response = _make_anthropic_text_response("Done")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("529 overloaded"),
            good_response,
        ]

        with patch.object(mb_mod._anthropic, "Anthropic", return_value=mock_client):
            with patch("agent_safety_eval.adapters.model_b.time.sleep"):
                adapter = mb_mod.AnthropicAdapter(api_key="fake-key")
                turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert turn.text == "Done"
        assert mock_client.messages.create.call_count == 2


# ===================================================================
# OpenAI adapter tests
# ===================================================================


class TestOpenAISchemaConversion:
    """Test conversion to OpenAI tool format."""

    def test_tools_to_openai(self) -> None:
        from agent_safety_eval.adapters.model_b import _tools_to_openai

        result = _tools_to_openai(SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["parameters"]["required"] == ["path"]


class TestOpenAIMessageConversion:
    """Test conversion to OpenAI message format."""

    def test_basic_messages(self) -> None:
        from agent_safety_eval.adapters.model_b import _messages_to_openai

        result = _messages_to_openai(SAMPLE_MESSAGES)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_assistant_with_tool_calls(self) -> None:
        from agent_safety_eval.adapters.model_b import _messages_to_openai

        messages = [
            {
                "role": "assistant",
                "content": "Sure.",
                "tool_calls": [
                    {"name": "read_file", "arguments": {"path": "/f.txt"}, "call_id": "tc-1"},
                ],
            },
        ]
        result = _messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Sure."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "tc-1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"
        # Arguments should be JSON-encoded
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"path": "/f.txt"}


class TestOpenAIResponseNormalization:
    """Test normalization of OpenAI responses."""

    def test_text_response(self) -> None:
        from agent_safety_eval.adapters.model_b import _openai_response_to_turn

        response = _make_openai_text_response("Here are the results.")
        turn = _openai_response_to_turn(response)

        assert turn.text == "Here are the results."
        assert turn.tool_calls == []

    def test_function_call_response(self) -> None:
        from agent_safety_eval.adapters.model_b import _openai_response_to_turn

        response = _make_openai_function_call_response(
            "read_file", {"path": "/notes.txt"}, "call_abc"
        )
        turn = _openai_response_to_turn(response)

        assert turn.text == ""
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert turn.tool_calls[0].arguments == {"path": "/notes.txt"}
        assert turn.tool_calls[0].call_id == "call_abc"


class TestOpenAIAdapter:
    """Test the OpenAIAdapter class."""

    def test_import_error_when_missing(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        original = mb_mod._HAS_OPENAI
        try:
            mb_mod._HAS_OPENAI = False
            with pytest.raises(ImportError, match="openai"):
                mb_mod.OpenAIAdapter(api_key="test-key")
        finally:
            mb_mod._HAS_OPENAI = original

    def test_missing_api_key(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_OPENAI:
            pytest.skip("openai not installed")

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    mb_mod.OpenAIAdapter()

    def test_dry_run(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_OPENAI:
            pytest.skip("openai not installed")

        with patch.object(mb_mod._openai, "OpenAI"):
            adapter = mb_mod.OpenAIAdapter(api_key="fake-key", dry_run=True)
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

            assert "[dry-run]" in turn.text

    def test_step_with_mocked_client(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_OPENAI:
            pytest.skip("openai not installed")

        mock_response = _make_openai_function_call_response(
            "send_email",
            {"to": "a@b.com", "subject": "Hi", "body": "Hello"},
            "call_xyz",
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(mb_mod._openai, "OpenAI", return_value=mock_client):
            adapter = mb_mod.OpenAIAdapter(api_key="fake-key")
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "send_email"
        assert turn.tool_calls[0].call_id == "call_xyz"
        assert adapter.total_input_tokens == 10
        assert adapter.total_output_tokens == 8

    def test_retry_on_transient_error(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_OPENAI:
            pytest.skip("openai not installed")

        good_response = _make_openai_text_response("OK")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("500 internal server error"),
            good_response,
        ]

        with patch.object(mb_mod._openai, "OpenAI", return_value=mock_client):
            with patch("agent_safety_eval.adapters.model_b.time.sleep"):
                adapter = mb_mod.OpenAIAdapter(api_key="fake-key")
                turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

        assert turn.text == "OK"
        assert mock_client.chat.completions.create.call_count == 2


# ===================================================================
# Model B factory tests
# ===================================================================


class TestModelBFactory:
    """Test the create_model_b_adapter factory function."""

    def test_invalid_format(self) -> None:
        from agent_safety_eval.adapters.model_b import create_model_b_adapter

        with pytest.raises(ValueError, match="provider:model_name"):
            create_model_b_adapter("just-a-model-name")

    def test_unknown_provider(self) -> None:
        from agent_safety_eval.adapters.model_b import create_model_b_adapter

        with pytest.raises(ValueError, match="Unknown provider"):
            create_model_b_adapter("cohere:command-r")

    def test_anthropic_creation(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_ANTHROPIC:
            pytest.skip("anthropic not installed")

        with patch.object(mb_mod._anthropic, "Anthropic"):
            adapter = mb_mod.create_model_b_adapter(
                "anthropic:claude-sonnet-4-20250514", api_key="fake-key"
            )
            assert isinstance(adapter, mb_mod.AnthropicAdapter)
            assert adapter.model == "claude-sonnet-4-20250514"

    def test_openai_creation(self) -> None:
        import agent_safety_eval.adapters.model_b as mb_mod

        if not mb_mod._HAS_OPENAI:
            pytest.skip("openai not installed")

        with patch.object(mb_mod._openai, "OpenAI"):
            adapter = mb_mod.create_model_b_adapter(
                "openai:gpt-4o", api_key="fake-key"
            )
            assert isinstance(adapter, mb_mod.OpenAIAdapter)
            assert adapter.model == "gpt-4o"


# ===================================================================
# ADK adapter tests
# ===================================================================


class TestADKSchemaConversion:
    """Test conversion of our tools to ADK format."""

    def test_tools_to_adk(self) -> None:
        from agent_safety_eval.adapters.adk import _our_tools_to_adk_declarations

        result = _our_tools_to_adk_declarations(SAMPLE_TOOLS)

        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert result[0]["parameters"]["type"] == "object"
        assert result[1]["name"] == "send_email"


class TestADKResponseExtraction:
    """Test extraction of AssistantTurn from ADK events."""

    def test_text_event(self) -> None:
        from agent_safety_eval.adapters.adk import _extract_turn_from_events

        part = MagicMock()
        part.text = "Hello"
        part.function_call = None

        content = MagicMock()
        content.parts = [part]

        event = MagicMock()
        event.content = content

        turn = _extract_turn_from_events([event])
        assert turn.text == "Hello"
        assert turn.tool_calls == []

    def test_function_call_event(self) -> None:
        from agent_safety_eval.adapters.adk import _extract_turn_from_events

        fc = MagicMock()
        fc.name = "read_file"
        fc.args = {"path": "/notes.txt"}

        part = MagicMock()
        part.text = None
        part.function_call = fc

        content = MagicMock()
        content.parts = [part]

        event = MagicMock()
        event.content = content

        turn = _extract_turn_from_events([event])
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"
        assert turn.tool_calls[0].call_id.startswith("adk-")

    def test_empty_events(self) -> None:
        from agent_safety_eval.adapters.adk import _extract_turn_from_events

        turn = _extract_turn_from_events([])
        assert turn.text == ""
        assert turn.tool_calls == []


class TestADKAdapter:
    """Test the ADKAdapter class."""

    def test_import_error_when_missing(self) -> None:
        import agent_safety_eval.adapters.adk as adk_mod

        original = adk_mod._HAS_ADK
        try:
            adk_mod._HAS_ADK = False
            with pytest.raises(ImportError, match="google-adk"):
                adk_mod.ADKAdapter(api_key="test-key")
        finally:
            adk_mod._HAS_ADK = original

    def test_missing_api_key(self) -> None:
        import agent_safety_eval.adapters.adk as adk_mod

        if not adk_mod._HAS_ADK:
            pytest.skip("google-adk not installed")

        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("GEMINI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key"):
                    adk_mod.ADKAdapter()

    def test_dry_run(self) -> None:
        import agent_safety_eval.adapters.adk as adk_mod

        if not adk_mod._HAS_ADK:
            pytest.skip("google-adk not installed")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            adapter = adk_mod.ADKAdapter(api_key="fake-key", dry_run=True)
            turn = adapter.step(SAMPLE_MESSAGES, SAMPLE_TOOLS)

            assert "[dry-run]" in turn.text
            assert turn.tool_calls == []


# ===================================================================
# Cross-provider normalization consistency tests
# ===================================================================


class TestCrossProviderNormalization:
    """Verify that all adapters produce the same ToolCall format."""

    def test_tool_call_format_gemini(self) -> None:
        """Gemini adapter produces standard ToolCall objects."""
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn

        response = _make_gemini_function_call_response(
            "send_email",
            {"to": "x@y.com", "subject": "Hi", "body": "Hey"},
        )
        turn = _gemini_response_to_turn(response)

        tc = turn.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert isinstance(tc.name, str)
        assert isinstance(tc.arguments, dict)
        assert isinstance(tc.call_id, str)

    def test_tool_call_format_anthropic(self) -> None:
        """Anthropic adapter produces standard ToolCall objects."""
        from agent_safety_eval.adapters.model_b import _anthropic_response_to_turn

        response = _make_anthropic_tool_use_response(
            None, "send_email",
            {"to": "x@y.com", "subject": "Hi", "body": "Hey"},
            "toolu_123",
        )
        turn = _anthropic_response_to_turn(response)

        tc = turn.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert isinstance(tc.name, str)
        assert isinstance(tc.arguments, dict)
        assert isinstance(tc.call_id, str)

    def test_tool_call_format_openai(self) -> None:
        """OpenAI adapter produces standard ToolCall objects."""
        from agent_safety_eval.adapters.model_b import _openai_response_to_turn

        response = _make_openai_function_call_response(
            "send_email",
            {"to": "x@y.com", "subject": "Hi", "body": "Hey"},
            "call_abc",
        )
        turn = _openai_response_to_turn(response)

        tc = turn.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert isinstance(tc.name, str)
        assert isinstance(tc.arguments, dict)
        assert isinstance(tc.call_id, str)

    def test_all_adapters_same_fields(self) -> None:
        """All three providers normalize to the same ToolCall field set."""
        from agent_safety_eval.adapters.gemini import _gemini_response_to_turn
        from agent_safety_eval.adapters.model_b import (
            _anthropic_response_to_turn,
            _openai_response_to_turn,
        )

        gemini_tc = _gemini_response_to_turn(
            _make_gemini_function_call_response("read_file", {"path": "/f"})
        ).tool_calls[0]

        anthropic_tc = _anthropic_response_to_turn(
            _make_anthropic_tool_use_response(None, "read_file", {"path": "/f"}, "id1")
        ).tool_calls[0]

        openai_tc = _openai_response_to_turn(
            _make_openai_function_call_response("read_file", {"path": "/f"}, "id2")
        ).tool_calls[0]

        # All should have same name and arguments
        for tc in [gemini_tc, anthropic_tc, openai_tc]:
            assert tc.name == "read_file"
            assert tc.arguments == {"path": "/f"}
            assert hasattr(tc, "call_id")


# ===================================================================
# Live API tests (skipped without keys)
# ===================================================================


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestGeminiLive:
    """Live tests for Gemini adapter. Only run with real API key."""

    def test_simple_text_response(self) -> None:
        from agent_safety_eval.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        turn = adapter.step(
            [{"role": "user", "content": "Say hello in one word."}],
            [],
        )
        assert turn.text  # Should have some text


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicLive:
    """Live tests for Anthropic adapter. Only run with real API key."""

    def test_simple_text_response(self) -> None:
        from agent_safety_eval.adapters.model_b import AnthropicAdapter

        adapter = AnthropicAdapter()
        turn = adapter.step(
            [{"role": "user", "content": "Say hello in one word."}],
            [],
        )
        assert turn.text


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAILive:
    """Live tests for OpenAI adapter. Only run with real API key."""

    def test_simple_text_response(self) -> None:
        from agent_safety_eval.adapters.model_b import OpenAIAdapter

        adapter = OpenAIAdapter()
        turn = adapter.step(
            [{"role": "user", "content": "Say hello in one word."}],
            [],
        )
        assert turn.text
