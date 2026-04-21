from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import ManifestPayload


def validate_backend_artifacts(entry: dict[str, Any]) -> bool:
    if entry.get("status") != "success":
        return False

    structure_path = entry.get("structure_path")
    if not structure_path:
        return False

    try:
        return Path(str(structure_path)).exists()
    except (OSError, TypeError, ValueError):
        return False


def is_cached(manifest: ManifestPayload, backend_name: str, config_hash: str) -> bool:
    backend_state = manifest.get(backend_name, {})
    if not isinstance(backend_state, dict):
        return False
    if backend_state.get("status") != "success":
        return False
    if backend_state.get("config_hash") != config_hash:
        return False
    if not validate_backend_artifacts(backend_state):
        print(f"[CACHE INVALID] missing artifacts for backend={backend_name}")
        return False
    return True
