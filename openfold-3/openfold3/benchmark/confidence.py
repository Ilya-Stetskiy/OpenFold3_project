from __future__ import annotations

import json
from pathlib import Path


def load_confidence_json(confidence_path: Path) -> dict:
    payload = json.loads(confidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Confidence payload at {confidence_path} must be an object")
    return payload
