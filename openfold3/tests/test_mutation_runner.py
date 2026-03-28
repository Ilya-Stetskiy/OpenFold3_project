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

from openfold3.mutation_runner import (
    MutationScreeningRunner,
    MutationSpec,
    ScreeningJob,
    ScreeningResultRow,
    SubprocessOpenFoldBackend,
    apply_point_mutation,
    sequence_hash,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


class FakeBackend:
    def __init__(self):
        self.calls = 0
        self.payload_sequences = []

    def run(self, prepared_job):
        self.calls += 1
        payload = json.loads(prepared_job.payload_path.read_text(encoding="utf-8"))
        query = payload["queries"][prepared_job.query_id]
        self.payload_sequences.append(
            [chain.get("sequence") for chain in query["chains"] if chain.get("sequence")]
        )
        return ScreeningResultRow(
            mutation_id=prepared_job.mutation_id,
            query_id=prepared_job.query_id,
            query_hash=prepared_job.query_hash,
            sample_index=1,
            seed=42,
            sample_ranking_score=0.8,
            iptm=0.7,
            ptm=0.6,
            avg_plddt=0.9,
            gpde=1.2,
            has_clash=0.0,
            cache_hit=prepared_job.cache_hit,
            sequence_cache_hits=prepared_job.sequence_cache_hits,
            query_result_cache_hit=False,
            cpu_prep_seconds=prepared_job.cpu_prep_seconds,
            gpu_inference_seconds=0.01,
            total_seconds=prepared_job.cpu_prep_seconds + 0.01,
            output_dir=str(prepared_job.output_dir),
            aggregated_confidence_path=None,
            mutation_spec=(
                None
                if prepared_job.mutation_spec is None
                else {
                    "chain_id": prepared_job.mutation_spec.chain_id,
                    "position_1based": prepared_job.mutation_spec.position_1based,
                    "from_residue": prepared_job.mutation_spec.from_residue,
                    "to_residue": prepared_job.mutation_spec.to_residue,
                }
            ),
        )


def test_apply_point_mutation():
    assert apply_point_mutation("ACDE", 2, "G", expected_residue="C") == "AGDE"


def test_sequence_hash_is_stable():
    assert sequence_hash("ACDE") == sequence_hash("ACDE")
    assert sequence_hash("ACDE") != sequence_hash("AGDE")


def test_parse_best_summary_row_keeps_zero_metrics(tmp_path):
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "demo",
                        "sample_ranking_score": 0.0,
                        "iptm": 0.0,
                        "ptm": 0.0,
                        "avg_plddt": 0.0,
                        "gpde": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "demo",
                        "sample_ranking_score": -1.0,
                        "iptm": -1.0,
                        "ptm": -1.0,
                        "avg_plddt": -1.0,
                        "gpde": 5.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best = SubprocessOpenFoldBackend.__new__(SubprocessOpenFoldBackend)._parse_best_summary_row(
        summary_path, "demo"
    )

    assert best["sample_ranking_score"] == 0.0
    assert best["gpde"] == 0.0


def test_mutation_screening_runner_raises_for_unknown_mutation_chain(tmp_path):
    base_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"},
            ]
        }
    )
    job = ScreeningJob(
        base_query=base_query,
        mutations=[
            MutationSpec(chain_id="B", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    try:
        MutationScreeningRunner(backend=FakeBackend()).run(job)
    except ValueError as exc:
        assert "Could not find mutable protein chain B" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown mutation chain")


def test_mutation_screening_runner_uses_sequence_cache_and_result_resume(tmp_path):
    msa_dir = tmp_path / "alignments_B"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": ["B"],
                    "sequence": "BBBB",
                    "main_msa_file_paths": [str(msa_dir)],
                },
            ]
        }
    )

    job = ScreeningJob(
        base_query=base_query,
        mutations=[MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=2,
        max_inflight_queries=2,
    )
    backend = FakeBackend()
    runner = MutationScreeningRunner(backend=backend)

    first_rows = runner.run(job)
    assert backend.calls == 2
    assert len(first_rows) == 2
    assert any("AGDE" in seqs for seqs in backend.payload_sequences)
    assert all(row.sequence_cache_hits >= 1 for row in first_rows)
    assert (job.output_dir / "results.jsonl").exists()
    assert (job.output_dir / "results.csv").exists()
    assert (job.output_dir / "screening_manifest.json").exists()

    second_rows = runner.run(job)
    assert backend.calls == 2
    assert len(second_rows) == 2
    assert all(row.query_result_cache_hit for row in second_rows)
