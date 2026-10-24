import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rosetta_adapter = load_module("scripts/executors/rosetta_adapter.py", "rosetta_adapter")


def test_parse_single_site_mutation():
    assert rosetta_adapter.parse_single_site_mutation("D105A") == ("D", "105", "A")
    assert rosetta_adapter.parse_single_site_mutation("g30f") == ("G", "30", "F")


def test_invalid_mutation_raises():
    with pytest.raises(ValueError):
        rosetta_adapter.parse_single_site_mutation("D105A/G30F")


def test_write_mutation_file(tmp_path):
    path = rosetta_adapter.write_mutation_file(tmp_path, "A", "D105A")
    assert path.read_text(encoding="utf-8") == "total 1\n1\nD 105 A\n"


def test_parse_fake_rosetta_score_file(tmp_path):
    work = tmp_path / "rosetta_work"
    work.mkdir()
    (work / "score.sc").write_text("SCORE: total_score ddg description\nSCORE: -10.0 -1.23 model_1\n", encoding="utf-8")

    parsed = rosetta_adapter.parse_rosetta_ddg(tmp_path)

    assert parsed["parser_status"] == "success"
    assert parsed["rosetta_ddg"] == -1.23

