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

from openfold3.panel_stand import PanelDdgStandRunner, PanelStandConfig


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
