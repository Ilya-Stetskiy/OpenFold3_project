from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPERS_DIR = PROJECT_ROOT / "helpers"
DEFAULT_QUERY_JSON = PROJECT_ROOT / "server_smoke" / "query_ubiquitin.json"


def ensure_helpers_on_path() -> None:
    helpers_path = str(HELPERS_DIR)
    if helpers_path not in sys.path:
        sys.path.insert(0, helpers_path)


def load_query_molecules(
    query_json: Path | None = None,
    query_id: str | None = None,
) -> tuple[str, list[dict]]:
    payload = json.loads(
        (query_json or DEFAULT_QUERY_JSON).read_text(encoding="utf-8")
    )
    queries = payload["queries"]
    resolved_query_id = query_id or next(iter(queries))
    return resolved_query_id, queries[resolved_query_id]["chains"]
