from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BackendResult


ManifestPayload = dict[str, dict[str, Any]]


def load_manifest(path: Path) -> ManifestPayload:
    if not path.exists():
        return {"openfold3": {}, "foldx": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest payload at {path}")
    normalized: ManifestPayload = {"openfold3": {}, "foldx": {}}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def save_manifest(path: Path, manifest: ManifestPayload) -> None:
    normalized = {"openfold3": {}, "foldx": {}}
    normalized.update(manifest)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")


def manifest_entry_from_result(result: BackendResult, config_hash: str) -> dict[str, Any]:
    structure_path = str(result.artifact_paths[0]) if result.artifact_paths else None
    return {
        "status": "success" if result.status == "ok" else result.status,
        "config_hash": config_hash,
        "output_dir": str(result.output_dir),
        "artifact_paths": [str(path) for path in result.artifact_paths],
        "structure_path": structure_path,
        "message": result.message,
    }
