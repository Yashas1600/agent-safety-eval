"""Smoke tests for the FastMCP stdio server.

Starts mcp_server.py as a subprocess over stdio, lists tools, calls one tool,
and verifies the result. Uses the mcp client library (mcp<2).
"""

from __future__ import annotations

import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.fixture
def server_params() -> StdioServerParameters:
    """Server parameters to launch mcp_server.py over stdio."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_safety_eval.servers.mcp_server"],
    )


@pytest.mark.asyncio
async def test_list_tools(server_params: StdioServerParameters) -> None:
    """The MCP server should expose all 16 tools from the registry."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tool_names = sorted(t.name for t in result.tools)

            expected = sorted([
                "list_dir", "read_file", "write_file", "move_file",
                "delete_file", "empty_trash",
                "search_inbox", "read_email", "draft_email", "send_email",
                "delete_email", "archive_email",
                "list_events", "create_event", "update_event", "cancel_event",
            ])
            assert tool_names == expected, f"Got: {tool_names}"
            assert len(result.tools) == 16


@pytest.mark.asyncio
async def test_list_tools_annotations(server_params: StdioServerParameters) -> None:
    """Irreversible tools should have destructiveHint set to True."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

            destructive_tools = {
                t.name for t in result.tools
                if t.annotations and t.annotations.destructiveHint is True
            }
            expected_destructive = {
                "delete_file", "empty_trash",
                "send_email", "delete_email",
                "cancel_event",
            }
            assert destructive_tools == expected_destructive


@pytest.mark.asyncio
async def test_call_write_and_read_file(server_params: StdioServerParameters) -> None:
    """Write a file via MCP, then read it back and verify content."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Write a file
            write_result = await session.call_tool(
                "write_file",
                arguments={"path": "/hello.txt", "content": "world"},
            )
            assert write_result.content
            text = write_result.content[0].text
            assert '"ok"' in text

            # Read it back
            read_result = await session.call_tool(
                "read_file",
                arguments={"path": "/hello.txt"},
            )
            assert read_result.content
            text = read_result.content[0].text
            assert '"world"' in text


@pytest.mark.asyncio
async def test_call_list_dir_empty(server_params: StdioServerParameters) -> None:
    """Listing an empty directory should return an empty files list."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "list_dir",
                arguments={"path": "/"},
            )
            assert result.content
            text = result.content[0].text
            assert '"ok"' in text
            assert '"files"' in text


@pytest.mark.asyncio
async def test_call_send_email(server_params: StdioServerParameters) -> None:
    """Send an email and verify it returns a success response."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "send_email",
                arguments={
                    "to": "alice@example.com",
                    "subject": "Test",
                    "body": "Hello",
                },
            )
            assert result.content
            text = result.content[0].text
            assert '"sent": true' in text or '"sent":true' in text


@pytest.mark.asyncio
async def test_call_create_and_cancel_event(
    server_params: StdioServerParameters,
) -> None:
    """Create a calendar event, then cancel it."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Create event
            create_result = await session.call_tool(
                "create_event",
                arguments={
                    "title": "Standup",
                    "start": "2026-01-15T09:00:00",
                    "end": "2026-01-15T09:30:00",
                    "attendees": ["bob@example.com"],
                },
            )
            assert create_result.content
            text = create_result.content[0].text
            assert "event-1" in text

            # Cancel it
            cancel_result = await session.call_tool(
                "cancel_event",
                arguments={"id": "event-1"},
            )
            assert cancel_result.content
            text = cancel_result.content[0].text
            assert '"cancelled": true' in text or '"cancelled":true' in text
