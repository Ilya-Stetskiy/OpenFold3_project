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


foldx_adapter = load_module("scripts/executors/foldx_adapter.py", "foldx_adapter")


def test_foldx_mutation_code():
    assert foldx_adapter.build_foldx_mutation_code("A", "D105A") == "DA105A;"
    assert foldx_adapter.build_foldx_mutation_code("B", "G30F") == "GB30F;"


def test_invalid_mutation_raises():
    with pytest.raises(ValueError):
        foldx_adapter.build_foldx_mutation_code("A", "D105A/G30F")


def test_parse_fake_foldx_output(tmp_path):
    work = tmp_path / "foldx_work"
    work.mkdir()
    (work / "Dif_test.fxout").write_text("Pdb\tddG\nmodel_1\t-1.23\n", encoding="utf-8")

    parsed = foldx_adapter.parse_foldx_ddg(tmp_path)

    assert parsed["parser_status"] == "success"
    assert parsed["foldx_ddg"] == -1.23
    assert parsed["foldx_unit"] == "kcal/mol"

