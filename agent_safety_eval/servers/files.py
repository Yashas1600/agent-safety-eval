"""Mock files server operating on WorldState."""

from __future__ import annotations

from typing import Any

from agent_safety_eval.world import WorldState


def list_dir(world: WorldState, path: str = "") -> dict[str, Any]:
    """List files whose paths start with *path* (or all files if empty)."""
    prefix = path.rstrip("/")
    matches = [p for p in sorted(world.files) if p.startswith(prefix)]
    return {"status": "ok", "files": matches}


def read_file(world: WorldState, path: str) -> dict[str, Any]:
    """Return the content of a file."""
    if path not in world.files:
        return {"status": "error", "message": f"File not found: {path}"}
    return {"status": "ok", "path": path, "content": world.files[path]}


def write_file(world: WorldState, path: str, content: str) -> dict[str, Any]:
    """Create or overwrite a file."""
    world.files[path] = content
    return {"status": "ok", "path": path}


def move_file(world: WorldState, src: str, dst: str) -> dict[str, Any]:
    """Move (rename) a file from *src* to *dst*."""
    if src not in world.files:
        return {"status": "error", "message": f"File not found: {src}"}
    world.files[dst] = world.files.pop(src)
    return {"status": "ok", "src": src, "dst": dst}


def delete_file(world: WorldState, path: str) -> dict[str, Any]:
    """Move a file to trash. Irreversible in the domain model."""
    if path not in world.files:
        return {"status": "error", "message": f"File not found: {path}"}
    world.trash[path] = world.files.pop(path)
    return {"status": "ok", "path": path, "moved_to_trash": True}


def empty_trash(world: WorldState) -> dict[str, Any]:
    """Permanently destroy all trashed files. Irreversible."""
    count = len(world.trash)
    world.trash.clear()
    return {"status": "ok", "deleted_count": count}
