from __future__ import annotations

import sqlite3
from pathlib import Path

from of_notebook_lib.ddg_panel import (
    PanelVisualRow,
    build_panel_preview,
    load_panel_visual_rows,
    parse_positions_spec,
    render_panel_structure_comparison_html,
    resolve_positions,
)


def test_parse_positions_spec_and_resolve_positions() -> None:
    assert parse_positions_spec("1-3, 5, 7-8", sequence_length=10) == (
        1,
        2,
        3,
        5,
        7,
        8,
    )
    assert resolve_positions(
        positions_mode="all_chain_positions",
        positions_text="",
        sequence_length=4,
    ) == (1, 2, 3, 4)


def test_build_panel_preview_uses_single_mutable_chain() -> None:
    molecules = [
        {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GG"},
        {"molecule_type": "protein", "chain_ids": ["D"], "sequence": "DE"},
    ]

    preview_df = build_panel_preview(
        "barnase_barstar",
        molecules,
        mutable_chain_id="D",
        positions=(1, 2),
    )

    assert list(preview_df["wt_residue"]) == ["D", "E"]
    assert set(preview_df["mutant_count"]) == {19}


def test_load_panel_visual_rows_reads_wt_and_foldx_paths(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite"
    foldx_path = tmp_path / "foldx_model.pdb"
    mutant_path = tmp_path / "mutant_model.cif"
    wt_path = tmp_path / "wt_model.cif"
    for path in (foldx_path, mutant_path, wt_path):
        path.write_text("data_demo\n", encoding="utf-8")

    conn = sqlite3.connect(state_db)
    conn.executescript(
        """
        CREATE TABLE wt_baseline (
            target_id TEXT PRIMARY KEY,
            query_id TEXT NOT NULL,
            msa_status TEXT,
            predict_status TEXT,
            analysis_status TEXT,
            msa_dir TEXT,
            msa_query_json TEXT,
            predict_dir TEXT,
            structure_path TEXT,
            confidence_path TEXT,
            rosetta_score REAL,
            wt_report_path TEXT,
            last_error TEXT,
            updated_at TEXT
        );
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            panel_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            chain_id TEXT NOT NULL,
            position_1based INTEGER NOT NULL,
            from_residue TEXT NOT NULL,
            to_residue TEXT NOT NULL,
            predict_status TEXT,
            analysis_status TEXT,
            structure_path TEXT,
            confidence_path TEXT,
            report_path TEXT,
            rosetta_delta_vs_wt REAL,
            last_error TEXT,
            updated_at TEXT
        );
        CREATE TABLE method_results (
            job_id TEXT NOT NULL,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            score REAL,
            units TEXT,
            details_json TEXT NOT NULL,
            PRIMARY KEY(job_id, method)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO wt_baseline (
            target_id, query_id, structure_path
        ) VALUES (?, ?, ?)
        """,
        ("demo", "wt", str(wt_path)),
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, panel_id, target_id, query_id, chain_id, position_1based,
            from_residue, to_residue, structure_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("demo_D_D39A", "demo_D_D39", "demo", "demo_D_D39A", "D", 39, "D", "A", str(mutant_path)),
    )
    conn.execute(
        """
        INSERT INTO method_results (job_id, method, status, score, units, details_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "demo_D_D39A",
            "foldx",
            "ok",
            0.0,
            "kcal/mol",
            '{"mutant_model_path": "%s"}' % foldx_path,
        ),
    )
    conn.commit()
    conn.close()

    rows = load_panel_visual_rows(state_db)

    assert len(rows) == 1
    assert rows[0].wt_structure_path == wt_path
    assert rows[0].foldx_mutant_model_path == foldx_path


def test_render_panel_structure_comparison_html_handles_missing_files() -> None:
    html = render_panel_structure_comparison_html(
        PanelVisualRow(
            job_id="demo_D_D39A",
            panel_id="demo_D_D39",
            target_id="demo",
            chain_id="D",
            position_1based=39,
            from_residue="D",
            to_residue="A",
            mutant_query_structure_path=None,
            wt_structure_path=None,
            foldx_mutant_model_path=None,
        )
    )

    assert "WT complex" in html
    assert "FoldX local mutant" in html
    assert "OpenFold mutant complex" in html
