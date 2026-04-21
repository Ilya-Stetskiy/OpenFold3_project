from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from foldx_worker import cli


def _write_probe_script(path: Path) -> None:
    path.write_text("#!/bin/sh\necho 'FoldX help'\n", encoding="utf-8")
    path.chmod(0o755)


def _args(tmp_path: Path, **overrides):
    values = {
        "output_dir": tmp_path / "out",
        "cache_dir": tmp_path / "cache",
        "allow_missing_foldx": False,
        "probe_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_preflight_rejects_non_launchable_foldx(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    bad_foldx = tmp_path / "foldx"
    bad_foldx.write_text("not a linux executable\n", encoding="utf-8")
    bad_foldx.chmod(0o755)
    monkeypatch.setenv("FOLDX_BINARY", str(bad_foldx))

    assert cli.preflight(_args(tmp_path)) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["foldx_probe"]["status"] == "failed"
    assert "Exec format" in payload["foldx_probe"]["error"]


def test_preflight_accepts_launchable_foldx(tmp_path: Path, monkeypatch, capsys) -> None:
    foldx = tmp_path / "foldx"
    _write_probe_script(foldx)
    monkeypatch.setenv("FOLDX_BINARY", str(foldx))

    assert cli.preflight(_args(tmp_path)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["foldx_binary"] == str(foldx)
    assert payload["foldx_probe"]["status"] == "ok"


def test_run_payload_marks_partial_failure_and_writes_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foldx = tmp_path / "foldx"
    _write_probe_script(foldx)
    monkeypatch.setenv("FOLDX_BINARY", str(foldx))

    structure_path = tmp_path / "input.pdb"
    structure_path.write_text("END\n", encoding="utf-8")
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "mode": "explicit_mutations",
                "structure_path": str(structure_path),
                "case_id": "case",
                "mutations": [
                    {
                        "chain_id": "A",
                        "from_residue": "L",
                        "position_1based": 1,
                        "to_residue": "A",
                    },
                    {
                        "chain_id": "A",
                        "from_residue": "L",
                        "position_1based": 2,
                        "to_residue": "V",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_local_mutation_case(*, mutation, case_id, **_kwargs):
        if mutation.to_residue == "V":
            raise RuntimeError("boom")
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            case_id=case_id,
            mutation=mutation,
            local_edit_status="ok",
            failure_reason=None,
            runtime_seconds=0.01,
            mutant_structure_path=None,
            report_path=report_path,
        )

    monkeypatch.setattr(cli, "run_local_mutation_case", fake_run_local_mutation_case)

    exit_code = cli.run_payload(
        _args(
            tmp_path,
            payload=payload_path,
            output_dir=tmp_path / "results",
            cache_dir=tmp_path / "cache",
        )
    )

    manifest = json.loads(
        (tmp_path / "results" / "foldx_worker_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert manifest["status"] == "partial_failed"
    assert manifest["total_mutations"] == 2
    assert manifest["successful_mutations"] == 1
    assert manifest["failed_mutations"] == 1
    assert (tmp_path / "results" / "rows.csv").exists()


def test_panel_manifest_fails_when_no_mutation_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foldx = tmp_path / "foldx"
    _write_probe_script(foldx)
    monkeypatch.setenv("FOLDX_BINARY", str(foldx))

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "mode": "panel",
                "structure_path": str(tmp_path / "input.pdb"),
                "chain_id": "A",
                "positions": [1],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_foldx_panel(**kwargs):
        output_root = Path(kwargs["output_root"])
        summary_json_path = output_root / "summary.json"
        rows_csv_path = output_root / "rows.csv"
        ranking_csv_path = output_root / "ranking.csv"
        for path in (summary_json_path, rows_csv_path, ranking_csv_path):
            path.write_text("", encoding="utf-8")
        row = SimpleNamespace(local_edit_status="failed")
        return SimpleNamespace(
            output_root=output_root,
            rows=(row,),
            summary_json_path=summary_json_path,
            rows_csv_path=rows_csv_path,
            ranking_csv_path=ranking_csv_path,
        )

    monkeypatch.setattr(cli, "run_foldx_panel", fake_run_foldx_panel)

    exit_code = cli.run_payload(
        _args(
            tmp_path,
            payload=payload_path,
            output_dir=tmp_path / "results",
            cache_dir=tmp_path / "cache",
        )
    )

    manifest = json.loads(
        (tmp_path / "results" / "foldx_worker_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert manifest["status"] == "failed"
    assert manifest["successful_mutations"] == 0
