import json
import sys
from pathlib import Path

from scripts.dev import run_ddg_testbench, summarize_ddg_testbench

from .test_ddg_benchmark_harness import _write_minimal_pdb


def test_run_ddg_testbench_main_single_case(tmp_path, monkeypatch, capsys):
    structure_path = tmp_path / "mini.pdb"
    output_root = tmp_path / "out"
    _write_minimal_pdb(structure_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ddg_testbench.py",
            "--output-root",
            str(output_root),
            "--dataset-kind",
            "benchmark",
            "--case-id",
            "cli-case",
            "--structure",
            str(structure_path),
            "--mutation",
            "A:L1A",
            "--experimental-ddg",
            "-1.2",
        ],
    )

    run_ddg_testbench.main()
    stdout = capsys.readouterr().out

    assert "run_root=" in stdout
    assert (output_root / "run_manifest.json").exists()
    assert (output_root / "reports" / "cli-case.json").exists()


def test_summarize_ddg_testbench_main(tmp_path, monkeypatch, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "case_id": "report-1",
                "structure_path": str(tmp_path / "mini.pdb"),
                "confidence_path": None,
                "structure_summary": {
                    "atom_count": 10,
                    "residue_count": 2,
                    "chain_ids": ["A", "B"],
                    "chain_lengths": {"A": 1, "B": 1},
                    "inferred_chain_groups": [["A"], ["B"]],
                    "min_inter_chain_atom_distance": 3.5,
                    "interface_atom_contacts_5a": 12,
                    "interface_ca_contacts_8a": 1,
                    "chain_pair_min_distances": {"A-B": 3.5},
                },
                "results": [
                    {
                        "method": "foldx",
                        "status": "ok",
                        "score": -0.5,
                        "units": "kcal/mol",
                        "details": {"runtime_seconds": 0.1},
                    }
                ],
                "experimental_ddg": -0.7,
                "notes": None,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ddg_testbench.py",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )

    summarize_ddg_testbench.main()
    stdout = capsys.readouterr().out

    assert '"total_cases": 1' in stdout
    assert output_path.exists()
