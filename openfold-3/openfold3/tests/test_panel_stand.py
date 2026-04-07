# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from openfold3.panel_stand import PanelDdgStandRunner, PanelStandConfig
from openfold3.panel_profiling import PanelExperimentProfiler


def test_panel_stand_builds_19_mutants_per_position(tmp_path):
    wt_query_json = tmp_path / "wt_query.json"
    wt_query_json.write_text(
        json.dumps(
            {
                "seeds": [42],
                "queries": {
                    "wt": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = PanelStandConfig(
        target_id="demo",
        wt_query_json=wt_query_json,
        output_root=tmp_path / "out",
        mutable_chain_id="A",
        positions=(2,),
    )
    runner = PanelDdgStandRunner(config)
    try:
        wt_query_id, _, panels = runner._build_panels()
        assert wt_query_id == "wt"
        assert len(panels) == 1
        panel = panels[0]
        assert panel.panel_id == "demo_A_C2"
        assert len(panel.job_ids) == 19
        assert "demo_A_C2G" in panel.job_ids
        assert "demo_A_C2C" not in panel.job_ids
    finally:
        runner.close()


def test_panel_stand_state_upserts_panel_jobs(tmp_path):
    wt_query_json = tmp_path / "wt_query.json"
    wt_query_json.write_text(
        json.dumps(
            {
                "seeds": [42],
                "queries": {
                    "wt": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = PanelStandConfig(
        target_id="demo",
        wt_query_json=wt_query_json,
        output_root=tmp_path / "out",
        mutable_chain_id="A",
        positions=(2, 3),
    )
    runner = PanelDdgStandRunner(config)
    try:
        _, wt_query, panels = runner._build_panels()
        for panel in panels:
            runner.db.upsert_panel("demo", panel, runner._panel_dir(panel.panel_id))
            runner._write_panel_query_json(wt_query, panel)
        assert len(runner.db.list_panels()) == 2
        first_jobs = runner.db.list_jobs_for_panel("demo_A_C2")
        second_jobs = runner.db.list_jobs_for_panel("demo_A_D3")
        assert len(first_jobs) == 19
        assert len(second_jobs) == 19
        assert runner.db.fetch_job("demo_A_C2G") is not None
        assert (runner._panel_dir("demo_A_C2") / "queries.json").exists()
    finally:
        runner.close()


def test_panel_experiment_profiler_writes_artifacts(tmp_path):
    profiler = PanelExperimentProfiler(
        output_root=tmp_path / "profiling",
        run_id="demo-profile",
        sample_interval_seconds=0.01,
    )

    profiler.start()
    profiler.record_stage(
        "checkpoint_load",
        "start",
        source="pytest",
        details={"label": "synthetic"},
    )
    profiler.record_stage(
        "checkpoint_load",
        "end",
        source="pytest",
        details={"duration_seconds": 0.25},
    )
    artifacts = profiler.stop()

    assert artifacts.events_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.timeline_svg_path.exists()
    assert artifacts.samples_path.exists() or artifacts.samples_path.parent.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["event_count"] >= 2
    assert summary["wall_seconds"] >= 0.0


def test_panel_stand_reuses_wt_msa_for_mutants(tmp_path, monkeypatch):
    wt_query_json = tmp_path / "wt_query.json"
    wt_query_json.write_text(
        json.dumps(
            {
                "seeds": [42],
                "queries": {
                    "wt": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    msa_dir = tmp_path / "wt_msa_assets"
    msa_dir.mkdir()
    wt_query_msa_json = tmp_path / "wt_query_msa.json"
    wt_query_msa_json.write_text(
        json.dumps(
            {
                "seeds": [42],
                "queries": {
                    "wt": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                                "main_msa_file_paths": [str(msa_dir)],
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = PanelStandConfig(
        target_id="demo",
        wt_query_json=wt_query_json,
        output_root=tmp_path / "out",
        mutable_chain_id="A",
        positions=(2,),
        reuse_wt_msa_for_mutants=True,
    )
    runner = PanelDdgStandRunner(config)
    try:
        monkeypatch.setattr(
            runner,
            "_align_msa",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected align")),
        )
        _, wt_query, panels = runner._build_panels()
        panel = panels[0]
        runner.db.upsert_panel("demo", panel, runner._panel_dir(panel.panel_id))
        ok = runner._ensure_panel_msa(panel, wt_query, wt_query_msa_json=wt_query_msa_json)
        assert ok is True
        row = runner.db.fetch_panel(panel.panel_id)
        assert row is not None
        query_msa_path = Path(row["msa_query_json"])
        payload = json.loads(query_msa_path.read_text(encoding="utf-8"))
        mutant_query = payload["queries"]["demo_A_C2A"]
        assert mutant_query["chains"][0]["sequence"] == "AADE"
        assert mutant_query["chains"][0]["main_msa_file_paths"] == [str(msa_dir)]
    finally:
        runner.close()


def test_panel_stand_prepare_inputs_returns_prepare_payload(tmp_path, monkeypatch):
    wt_query_json = tmp_path / "wt_query.json"
    wt_query_json.write_text(
        json.dumps(
            {
                "seeds": [42],
                "queries": {
                    "wt": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = PanelStandConfig(
        target_id="demo",
        wt_query_json=wt_query_json,
        output_root=tmp_path / "out",
        mutable_chain_id="A",
        positions=(2, 3),
    )
    runner = PanelDdgStandRunner(config)
    try:
        def fake_ensure_wt_msa_only(wt_query_id, wt_query):
            runner.db.upsert_wt("demo", wt_query_id)
            wt_msa_dir = runner._wt_dir() / "msa"
            wt_msa_dir.mkdir(parents=True, exist_ok=True)
            wt_query_msa_json = wt_msa_dir / "query_msa.json"
            wt_query_msa_json.write_text(
                json.dumps(
                    {
                        "seeds": [42],
                        "queries": {
                            wt_query_id: {
                                "chains": [
                                    {
                                        "molecule_type": "protein",
                                        "chain_ids": ["A"],
                                        "sequence": "ACDE",
                                        "main_msa_file_paths": [str(wt_msa_dir)],
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner.db.set_wt_stage("demo", "msa", "done", msa_dir=wt_msa_dir, msa_query_json=wt_query_msa_json)
            return wt_query_msa_json

        monkeypatch.setattr(runner, "_ensure_wt_msa_only", fake_ensure_wt_msa_only)
        monkeypatch.setattr(
            runner,
            "_predict",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prepare must not predict")),
        )
        payload = runner.prepare_inputs()
        assert payload["run_mode"] == "prepare"
        assert payload["panel_count"] == 2
        assert len(runner.db.list_panels()) == 2
        for row in runner.db.list_panels():
            assert row["msa_status"] == "done"
    finally:
        runner.close()
