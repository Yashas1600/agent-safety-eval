"""Conformance tests: run the same tool-call sequence against the Python
and Go MCP files servers, assert identical results.

Skipped when the ``go`` binary is not available on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

GO_BIN = shutil.which("go")
_go_available = GO_BIN is not None

needs_go = pytest.mark.skipif(not _go_available, reason="go not installed")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GO_SERVER_DIR = REPO_ROOT / "servers_go" / "files"


# ---------------------------------------------------------------------------
# Build the Go server once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def go_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the Go files server and return the path to the binary."""
    # Ensure dependencies are resolved
    subprocess.run(
        ["go", "mod", "tidy"],
        cwd=str(GO_SERVER_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    out_dir = tmp_path_factory.mktemp("go_build")
    binary = out_dir / "files-server-go"
    result = subprocess.run(
        ["go", "build", "-o", str(binary), "."],
        cwd=str(GO_SERVER_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Go build failed (skipping):\nstderr: {result.stderr}"
        )
    assert binary.exists(), "Go binary was not produced"
    return binary


# ---------------------------------------------------------------------------
# Tool-call script (same sequence for both servers)
# ---------------------------------------------------------------------------

TOOL_CALL_SCRIPT: list[tuple[str, dict[str, Any]]] = [
    # 1. Write two files.
    ("write_file", {"path": "/docs/a.txt", "content": "alpha"}),
    ("write_file", {"path": "/docs/b.txt", "content": "beta"}),
    ("write_file", {"path": "/other.txt", "content": "gamma"}),
    # 2. List all files.
    ("list_dir", {"path": ""}),
    # 3. List only /docs.
    ("list_dir", {"path": "/docs"}),
    # 4. Read a file.
    ("read_file", {"path": "/docs/a.txt"}),
    # 5. Read a nonexistent file.
    ("read_file", {"path": "/nope.txt"}),
    # 6. Move a file.
    ("move_file", {"src": "/docs/b.txt", "dst": "/archive/b.txt"}),
    # 7. Move a nonexistent file.
    ("move_file", {"src": "/gone.txt", "dst": "/x.txt"}),
    # 8. Delete a file (irreversible).
    ("delete_file", {"path": "/docs/a.txt"}),
    # 9. Delete a nonexistent file.
    ("delete_file", {"path": "/nope.txt"}),
    # 10. Empty trash.
    ("empty_trash", {}),
    # 11. Empty trash again (should be 0).
    ("empty_trash", {}),
    # 12. List remaining files.
    ("list_dir", {"path": ""}),
]


async def run_script_against_server(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Start an MCP server subprocess, run the tool-call script, return results."""
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )

    results: list[dict[str, Any]] = []

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            for tool_name, arguments in TOOL_CALL_SCRIPT:
                result = await session.call_tool(tool_name, arguments=arguments)
                assert len(result.content) > 0, (
                    f"No content in result for {tool_name}"
                )
                text = result.content[0].text
                parsed = json.loads(text)
                results.append(parsed)

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoConformance:
    """Run the same tool-call script against Python and Go servers."""

    @needs_go
    async def test_results_match(self, go_binary: Path) -> None:
        """Both servers produce identical results for the same script."""
        python_results = await run_script_against_server(
            command=sys.executable,
            args=["-m", "agent_safety_eval.servers.mcp_server"],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        go_results = await run_script_against_server(
            command=str(go_binary),
            args=[],
        )

        assert len(python_results) == len(TOOL_CALL_SCRIPT)
        assert len(go_results) == len(TOOL_CALL_SCRIPT)

        for i, (tool_name, _args) in enumerate(TOOL_CALL_SCRIPT):
            py_r = python_results[i]
            go_r = go_results[i]
            assert py_r == go_r, (
                f"Mismatch at step {i} ({tool_name}):\n"
                f"  Python: {json.dumps(py_r, sort_keys=True)}\n"
                f"  Go:     {json.dumps(go_r, sort_keys=True)}"
            )

    @needs_go
    async def test_irreversible_tools_exposed(self, go_binary: Path) -> None:
        """Go server exposes delete_file and empty_trash with destructiveHint."""
        server_params = StdioServerParameters(
            command=str(go_binary), args=[]
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools_by_name = {t.name: t for t in tools_result.tools}

                # All six tools must be present.
                expected = {
                    "list_dir", "read_file", "write_file",
                    "move_file", "delete_file", "empty_trash",
                }
                assert set(tools_by_name.keys()) == expected

                # Destructive hints on irreversible tools.
                for name in ("delete_file", "empty_trash"):
                    tool = tools_by_name[name]
                    assert tool.annotations is not None, (
                        f"{name} missing annotations"
                    )
                    assert tool.annotations.destructiveHint is True, (
                        f"{name} should have destructiveHint=true"
                    )

                # Non-destructive tools should not have destructiveHint.
                for name in ("list_dir", "read_file", "write_file", "move_file"):
                    tool = tools_by_name[name]
                    if tool.annotations is not None:
                        hint = tool.annotations.destructiveHint
                        assert hint is not True, (
                            f"{name} should not have destructiveHint=true"
                        )

    async def test_python_server_tools_listed(self) -> None:
        """Python MCP server exposes all six file tools (among others)."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent_safety_eval.servers.mcp_server"],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tool_names = {t.name for t in tools_result.tools}

                file_tools = {
                    "list_dir", "read_file", "write_file",
                    "move_file", "delete_file", "empty_trash",
                }
                assert file_tools.issubset(tool_names)
