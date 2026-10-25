"""FastMCP stdio server wrapping all mock-server tools (files, email, calendar).

Run as: python -m agent_safety_eval.servers.mcp_server

This exposes the same tools as the in-process registry but over real MCP stdio
transport, proving protocol compatibility and enabling --transport stdio runs.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from agent_safety_eval.servers.registry import TOOL_SCHEMAS, dispatch
from agent_safety_eval.world import WorldState

# Single global WorldState for the lifetime of the server process.
# Each stdio server is spawned per episode, so this is episode-scoped.
_world = WorldState()

mcp = FastMCP(
    name="agent-safety-eval",
    instructions="Mock MCP server for files, email, and calendar.",
)


def _annotations_for(schema: dict[str, Any]) -> ToolAnnotations:
    """Build MCP ToolAnnotations from our schema metadata."""
    irreversible = schema.get("irreversible", False)
    return ToolAnnotations(
        destructiveHint=irreversible,
        readOnlyHint=(not irreversible and schema.get("side_effect_scope") == "local"
                      and schema["description"].lower().startswith(("list", "read", "search"))),
        idempotentHint=False,
        openWorldHint=(schema.get("side_effect_scope") == "external"),
    )


def _make_handler(tool_name: str):
    """Create a handler closure for a given tool name."""
    def handler(**kwargs: Any) -> str:
        result = dispatch(tool_name, kwargs, _world)
        return json.dumps(result)
    # Set a useful function name for FastMCP introspection
    handler.__name__ = tool_name
    handler.__qualname__ = tool_name
    return handler


def _build_parameter_description(schema: dict[str, Any]) -> str:
    """Build a full description including irreversibility note."""
    desc = schema["description"]
    if schema.get("irreversible"):
        if "irreversible" not in desc.lower():
            desc += " [IRREVERSIBLE]"
    return desc


# Register all tools from the registry
for _name, _schema in TOOL_SCHEMAS.items():
    _handler = _make_handler(_name)

    # Build the parameter annotations dynamically by inspecting the schema.
    # FastMCP expects a callable with typed parameters matching the JSON schema.
    # We register using add_tool for full control.
    _desc = _build_parameter_description(_schema)
    _annot = _annotations_for(_schema)

    # We need to set __doc__ and proper annotations for FastMCP to pick up params.
    # Instead, use add_tool with fn_params built from schema properties.
    _props = _schema["parameters"].get("properties", {})
    _required = set(_schema["parameters"].get("required", []))

    # Build function with proper signature using exec
    # This is necessary because FastMCP inspects function signatures
    _param_parts = []
    _param_annotations = {}
    for _pname, _pdef in _props.items():
        _ptype = _pdef.get("type", "string")
        if _ptype == "string":
            _param_annotations[_pname] = str
            if _pname in _required:
                _param_parts.append(f"{_pname}: str")
            else:
                _param_parts.append(f'{_pname}: str = ""')
        elif _ptype == "array":
            _param_annotations[_pname] = list
            if _pname in _required:
                _param_parts.append(f"{_pname}: list")
            else:
                _param_parts.append(f"{_pname}: list = []")
        elif _ptype == "object":
            _param_annotations[_pname] = dict
            if _pname in _required:
                _param_parts.append(f"{_pname}: dict")
            else:
                _param_parts.append(f"{_pname}: dict = {{}}")
        else:
            _param_annotations[_pname] = str
            if _pname in _required:
                _param_parts.append(f"{_pname}: str")
            else:
                _param_parts.append(f'{_pname}: str = ""')

    # Build a function with the right signature
    _sig = ", ".join(_param_parts)
    _func_code = f"def {_name}({_sig}) -> str: return _inner(**{{k: v for k, v in locals().items()}})"
    _ns: dict[str, Any] = {"_inner": _handler}
    exec(_func_code, _ns)  # noqa: S102
    _typed_handler = _ns[_name]
    _typed_handler.__doc__ = _desc

    mcp.tool(
        name=_name,
        description=_desc,
        annotations=_annot,
    )(_typed_handler)


def get_world() -> WorldState:
    """Access the server's WorldState (useful for testing)."""
    return _world


def reset_world() -> None:
    """Reset the global WorldState to a fresh instance."""
    global _world
    _world = WorldState()


if __name__ == "__main__":
    mcp.run(transport="stdio")
