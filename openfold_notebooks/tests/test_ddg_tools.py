from __future__ import annotations

import os
from pathlib import Path

from of_notebook_lib.ddg_tools import collect_ddg_tool_status, export_ddg_tool_env


def test_collect_ddg_tool_status_prefers_repo_tools(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tools" / "bin").mkdir(parents=True)
    (tmp_path / "tools" / "bin" / "foldx").write_text("", encoding="utf-8")
    rosetta_bin = tmp_path / "tools" / "rosetta3.15_min" / "rosetta.binary.ubuntu.release-408" / "main" / "source" / "bin"
    rosetta_bin.mkdir(parents=True)
    (tmp_path / "tools" / "bin" / "score_jd2").write_text("", encoding="utf-8")
    (rosetta_bin / "score_jd2.static.linuxgccrelease").write_text("", encoding="utf-8")
    database = tmp_path / "tools" / "rosetta3.15_min" / "rosetta.binary.ubuntu.release-408" / "main" / "database"
    database.mkdir(parents=True)

    monkeypatch.delenv("FOLDX_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_SCORE_JD2_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_DATABASE", raising=False)

    statuses = collect_ddg_tool_status(tmp_path)

    assert statuses["foldx"].found is True
    assert statuses["foldx"].path == str((tmp_path / "tools" / "bin" / "foldx").resolve())
    assert statuses["rosetta_score_jd2"].found is True
    assert statuses["rosetta_score_jd2"].path == str((tmp_path / "tools" / "bin" / "score_jd2").resolve())
    assert statuses["rosetta_database"].found is True
    assert statuses["rosetta_database"].path == str(database.resolve())


def test_collect_ddg_tool_status_finds_foldx_in_local_foldx_dir(tmp_path: Path, monkeypatch) -> None:
    foldx_bin = tmp_path / "foldx" / "foldx5" / "foldx_20270131"
    foldx_bin.parent.mkdir(parents=True)
    foldx_bin.write_text("", encoding="utf-8")

    monkeypatch.delenv("FOLDX_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_SCORE_JD2_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_DATABASE", raising=False)

    statuses = collect_ddg_tool_status(tmp_path)

    assert statuses["foldx"].found is True
    assert statuses["foldx"].path == str(foldx_bin.resolve())
    assert statuses["foldx"].source == "repo:foldx/**"


def test_collect_ddg_tool_status_finds_rosetta_bundle_layout(tmp_path: Path, monkeypatch) -> None:
    bundle_root = tmp_path / "tools" / "rosetta3.15_min" / "rosetta.binary.ubuntu.release-408" / "main"
    score_jd2 = bundle_root / "source" / "bin" / "score_jd2.static.linuxgccrelease"
    score_jd2.parent.mkdir(parents=True)
    score_jd2.write_text("", encoding="utf-8")
    database = bundle_root / "database"
    database.mkdir(parents=True)

    monkeypatch.delenv("ROSETTA_SCORE_JD2_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_DATABASE", raising=False)

    statuses = collect_ddg_tool_status(tmp_path)

    assert statuses["rosetta_score_jd2"].found is True
    assert statuses["rosetta_score_jd2"].path == str(score_jd2.resolve())
    assert statuses["rosetta_database"].found is True
    assert statuses["rosetta_database"].path == str(database.resolve())


def test_export_ddg_tool_env_sets_expected_vars(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tools" / "bin").mkdir(parents=True)
    foldx = tmp_path / "tools" / "bin" / "foldx"
    score_jd2 = tmp_path / "tools" / "bin" / "score_jd2"
    database = tmp_path / "tools" / "rosetta" / "database"
    for path in (foldx, score_jd2):
        path.write_text("", encoding="utf-8")
    database.mkdir(parents=True)

    monkeypatch.delenv("FOLDX_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_SCORE_JD2_BINARY", raising=False)
    monkeypatch.delenv("ROSETTA_DATABASE", raising=False)

    exported = export_ddg_tool_env(tmp_path)

    assert exported["FOLDX_BINARY"] == str(foldx.resolve())
    assert exported["ROSETTA_SCORE_JD2_BINARY"] == str(score_jd2.resolve())
    assert exported["ROSETTA_DATABASE"] == str(database.resolve())
    assert os.environ["FOLDX_BINARY"] == exported["FOLDX_BINARY"]
