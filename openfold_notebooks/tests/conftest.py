from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPERS_DIR = PROJECT_ROOT / "helpers"

if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_root(project_root: Path) -> Path:
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def big_ace_output_dir(fixtures_root: Path) -> Path:
    return fixtures_root / "openfold_output" / "output"


@pytest.fixture(scope="session")
def big_ace_query_path(big_ace_output_dir: Path) -> Path:
    return big_ace_output_dir / "inference_query_set.json"


@pytest.fixture(scope="session")
def big_ace_query_payload(big_ace_query_path: Path) -> dict:
    with big_ace_query_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def big_ace_molecules(big_ace_query_payload: dict) -> list[dict]:
    return big_ace_query_payload["queries"]["complex_AB"]["chains"]
