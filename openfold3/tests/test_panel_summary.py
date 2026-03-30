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

from openfold3.panel_stand import MutationPanel, PanelStandState
from openfold3.benchmark.models import MutationInput
from openfold3.panel_summary import summarize_panel_state_db, write_panel_summary_outputs


def test_panel_summary_builds_consensus_and_exports(tmp_path):
    state = PanelStandState(tmp_path / "state.sqlite")
    try:
        state.upsert_wt("demo", "wt")
        state.set_wt_stage(
            "demo",
            "analysis",
            "done",
            rosetta_score=100.0,
        )
        panel = MutationPanel(
            panel_id="demo_A_D614",
            chain_id="A",
            position_1based=614,
            from_residue="D",
            job_ids=("demo_A_D614A", "demo_A_D614F"),
            mutations=(
                MutationInput("A", "D", 614, "A"),
                MutationInput("A", "D", 614, "F"),
            ),
        )
        panel_dir = tmp_path / "panels" / panel.panel_id
        state.upsert_panel("demo", panel, panel_dir)
        state.update_job_predict(
            "demo_A_D614A",
            status="done",
            structure_path=tmp_path / "a.cif",
            confidence_path=tmp_path / "a.json",
        )
        state.update_job_predict(
            "demo_A_D614F",
            status="done",
            structure_path=tmp_path / "f.cif",
            confidence_path=tmp_path / "f.json",
        )
        state.replace_method_results(
            "demo_A_D614A",
            [
                {"method": "foldx", "status": "ok", "score": 0.1, "units": "kcal/mol", "details": {}},
                {"method": "saambe_3d", "status": "ok", "score": 0.2, "units": "kcal/mol", "details": {}},
                {"method": "helixon_binding_ddg", "status": "ok", "score": 0.3, "units": "kcal/mol", "details": {}},
                {"method": "prompt_ddg", "status": "ok", "score": 0.4, "units": "kcal/mol", "details": {}},
                {"method": "rosetta_score", "status": "ok", "score": 101.0, "units": "reu", "details": {}},
            ],
        )
        state.replace_method_results(
            "demo_A_D614F",
            [
                {"method": "foldx", "status": "ok", "score": 0.5, "units": "kcal/mol", "details": {}},
                {"method": "saambe_3d", "status": "ok", "score": 0.6, "units": "kcal/mol", "details": {}},
                {"method": "helixon_binding_ddg", "status": "ok", "score": 0.7, "units": "kcal/mol", "details": {}},
                {"method": "prompt_ddg", "status": "ok", "score": 0.8, "units": "kcal/mol", "details": {}},
                {"method": "rosetta_score", "status": "ok", "score": 105.0, "units": "reu", "details": {}},
            ],
        )
        state.update_job_analysis(
            "demo_A_D614A",
            status="done",
            report_path=tmp_path / "a_report.json",
            rosetta_delta_vs_wt=1.0,
        )
        state.update_job_analysis(
            "demo_A_D614F",
            status="done",
            report_path=tmp_path / "f_report.json",
            rosetta_delta_vs_wt=5.0,
        )

        summary = summarize_panel_state_db(tmp_path / "state.sqlite")
        outputs = write_panel_summary_outputs(summary, tmp_path / "summary")

        assert summary.total_jobs == 2
        assert summary.analyzed_jobs == 2
        assert "rosetta_delta_vs_wt" in summary.methods
        assert summary.rows[0].consensus_z is not None
        assert summary.top_consensus[0]["job_id"] == "demo_A_D614F"
        assert outputs["summary_json"].exists()
        assert outputs["rows_csv"].exists()
    finally:
        state.close()
