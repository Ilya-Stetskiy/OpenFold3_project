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
import csv
import time
from pathlib import Path

from openfold3.mutation_runner import (
    PreparedMutationJob,
    MutationScreeningRunner,
    MutationSpec,
    QueryResultCache,
    ScreeningJob,
    ScreeningResultRow,
    SequenceArtifactCache,
    SubprocessOpenFoldBackend,
    apply_point_mutation,
    sequence_hash,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


def _result_row_from_prepared_job(prepared_job) -> ScreeningResultRow:
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
        return _result_row_from_prepared_job(prepared_job)


class BatchRecordingBackend:
    def __init__(self):
        self.batch_sizes = []
        self.query_id_batches = []
        self.run_calls = 0

    def run(self, prepared_job):
        self.run_calls += 1
        raise AssertionError("run() should not be used when run_batch() is available")

    def run_batch(self, prepared_jobs):
        self.batch_sizes.append(len(prepared_jobs))
        self.query_id_batches.append([job.query_id for job in prepared_jobs])
        return [
            _result_row_from_prepared_job(prepared_job)
            for prepared_job in prepared_jobs
        ]


class SlowPrepareRunner(MutationScreeningRunner):
    def __init__(self, delays_by_mutation_id, backend):
        super().__init__(backend=backend)
        self.delays_by_mutation_id = delays_by_mutation_id

    def _prepare_single_job(
        self,
        job,
        mutation_spec,
        sequence_cache,
        result_cache,
        mutable_chain_ids,
    ):
        mutation_id = "WT" if mutation_spec is None else mutation_spec.mutation_id
        delay_seconds = self.delays_by_mutation_id.get(mutation_id, 0.0)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return super()._prepare_single_job(
            job,
            mutation_spec,
            sequence_cache,
            result_cache,
            mutable_chain_ids,
        )


def test_apply_point_mutation():
    assert apply_point_mutation("ACDE", 2, "G", expected_residue="C") == "AGDE"


def test_sequence_hash_is_stable():
    assert sequence_hash("ACDE") == sequence_hash("ACDE")
    assert sequence_hash("ACDE") != sequence_hash("AGDE")


def test_mutation_spec_mutation_id_is_uppercase_and_stable():
    mutation = MutationSpec(
        chain_id="A", position_1based=2, from_residue="c", to_residue="g"
    )

    assert mutation.mutation_id == "A_C2G"


def test_apply_point_mutation_returns_same_sequence_for_noop_change():
    assert apply_point_mutation("ACDE", 2, "C", expected_residue="C") == "ACDE"


def test_apply_point_mutation_rejects_out_of_bounds_position():
    try:
        apply_point_mutation("ACDE", 5, "G", expected_residue="E")
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-bounds mutation")


def test_apply_point_mutation_rejects_unexpected_residue():
    try:
        apply_point_mutation("ACDE", 2, "G", expected_residue="A")
    except ValueError as exc:
        assert "Expected residue A" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched source residue")


def test_apply_point_mutation_rejects_noncanonical_residue():
    try:
        apply_point_mutation("ACDE", 2, "Z", expected_residue="C")
    except ValueError as exc:
        assert "canonical amino acid" in str(exc)
    else:
        raise AssertionError("Expected ValueError for noncanonical target residue")


def test_entries_respects_include_wt_and_run_baseline_first():
    job_first = ScreeningJob(
        base_query=Query.model_validate(
            {
                "chains": [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"}
                ]
            }
        ),
        mutations=[
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=Path("unused-output"),
        cache_dir=Path("unused-cache"),
        include_wt=True,
        run_baseline_first=True,
    )
    job_last = ScreeningJob(
        base_query=job_first.base_query,
        mutations=job_first.mutations,
        output_dir=Path("unused-output"),
        cache_dir=Path("unused-cache"),
        include_wt=True,
        run_baseline_first=False,
    )
    job_none = ScreeningJob(
        base_query=job_first.base_query,
        mutations=job_first.mutations,
        output_dir=Path("unused-output"),
        cache_dir=Path("unused-cache"),
        include_wt=False,
    )

    runner = MutationScreeningRunner()

    assert runner._entries(job_first) == [None, job_first.mutations[0]]
    assert runner._entries(job_last) == [job_last.mutations[0], None]
    assert runner._entries(job_none) == [job_none.mutations[0]]


def test_query_hash_changes_when_job_settings_change():
    query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"}
            ]
        }
    )
    runner = MutationScreeningRunner()
    base_job = ScreeningJob(
        base_query=query,
        mutations=[],
        output_dir=Path("unused-output"),
        cache_dir=Path("unused-cache"),
        num_diffusion_samples=1,
        num_model_seeds=1,
    )
    changed_job = ScreeningJob(
        base_query=query,
        mutations=[],
        output_dir=Path("unused-output"),
        cache_dir=Path("unused-cache"),
        num_diffusion_samples=4,
        num_model_seeds=1,
    )

    assert runner._query_hash(query, base_job) != runner._query_hash(query, changed_job)


def test_sequence_artifact_cache_round_trips_chain_paths(tmp_path):
    msa_dir = tmp_path / "alignments"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nACDE\n", encoding="utf-8")

    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                    "main_msa_file_paths": [str(msa_dir)],
                }
            ]
        }
    )
    cache = SequenceArtifactCache(tmp_path / "cache")
    cache.register_chain(query.chains[0])

    target_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"}
            ]
        }
    )

    assert cache.resolve_chain(target_query.chains[0]) is True
    assert [str(path) for path in target_query.chains[0].main_msa_file_paths] == [str(msa_dir)]


def test_query_result_cache_round_trips_row(tmp_path):
    cache = QueryResultCache(tmp_path / "results")
    row = ScreeningResultRow(
        mutation_id="WT",
        query_id="query_WT",
        query_hash="hash123",
        sample_index=1,
        seed=42,
        sample_ranking_score=0.8,
        iptm=0.7,
        ptm=0.6,
        avg_plddt=0.9,
        gpde=1.2,
        has_clash=0.0,
        cache_hit=False,
        sequence_cache_hits=0,
        query_result_cache_hit=False,
        cpu_prep_seconds=0.01,
        gpu_inference_seconds=0.02,
        total_seconds=0.03,
        output_dir=str(tmp_path / "output"),
        aggregated_confidence_path=None,
        mutation_spec=None,
    )

    cache.store(row)
    loaded = cache.load("hash123")

    assert loaded is not None
    assert loaded.query_id == row.query_id
    assert loaded.total_seconds == row.total_seconds


def test_parse_best_summary_row_preserves_zero_metric_values(tmp_path):
    job = ScreeningJob(
        base_query=Query.model_validate(
            {
                "chains": [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"}
                ]
            }
        ),
        mutations=[],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
    )
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "job_A_C2G",
                        "sample_index": 1,
                        "sample_ranking_score": 0.0,
                        "iptm": 0.0,
                        "ptm": 0.0,
                        "avg_plddt": 0.0,
                        "gpde": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "job_A_C2G",
                        "sample_index": 2,
                        "sample_ranking_score": None,
                        "iptm": None,
                        "ptm": None,
                        "avg_plddt": None,
                        "gpde": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best = SubprocessOpenFoldBackend(job)._parse_best_summary_row(summary_path, "job_A_C2G")

    assert best["sample_index"] == 1
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


def test_results_jsonl_and_csv_have_matching_row_counts(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    MutationScreeningRunner(backend=FakeBackend()).run(job)
    jsonl_rows = [
        json.loads(line)
        for line in (job.output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (job.output_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(jsonl_rows) == len(csv_rows) == 2


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


def test_mutation_screening_runner_does_not_cache_mutable_chain(tmp_path):
    msa_dir = tmp_path / "alignments"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                    "main_msa_file_paths": [str(msa_dir)],
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
        mutations=[
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="C")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    row = MutationScreeningRunner(backend=FakeBackend()).run(job)[0]

    assert row.sequence_cache_hits == 1


def test_mutation_screening_runner_marks_all_aliases_of_mutable_chain(tmp_path):
    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A", "A_copy"],
                    "sequence": "ACDE",
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": ["B"],
                    "sequence": "BBBB",
                },
            ]
        }
    )

    runner = MutationScreeningRunner()
    mutable_chain_ids = runner._mutable_chain_ids_from_query(
        base_query,
        [MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")],
    )

    assert mutable_chain_ids == {"A", "A_copy"}


def test_mutation_screening_runner_returns_empty_mutable_chain_ids_for_no_mutations():
    base_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"},
                {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "BBBB"},
            ]
        }
    )

    mutable_chain_ids = MutationScreeningRunner()._mutable_chain_ids_from_query(
        base_query, []
    )

    assert mutable_chain_ids == set()


def test_register_base_chain_cache_skips_mutable_chain_sequences(tmp_path):
    msa_dir = tmp_path / "alignments"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                    "main_msa_file_paths": [str(msa_dir)],
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

    cache = SequenceArtifactCache(tmp_path / "cache")
    runner = MutationScreeningRunner()
    runner._register_base_chain_cache(base_query, cache, {"A"})

    assert set(cache.index) == {sequence_hash("BBBB")}


def test_resolve_sequence_cache_counts_only_invariant_chain_hits(tmp_path):
    msa_dir = tmp_path / "alignments"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                    "main_msa_file_paths": [str(msa_dir)],
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

    cache = SequenceArtifactCache(tmp_path / "cache")
    cache.register_chain(query.chains[0])
    cache.register_chain(query.chains[1])

    hits = MutationScreeningRunner()._resolve_sequence_cache(query, cache, {"A"})

    assert hits == 1


def test_mutation_screening_runner_uses_all_chain_cache_when_no_mutations(tmp_path):
    msa_a = tmp_path / "alignments_A"
    msa_b = tmp_path / "alignments_B"
    msa_a.mkdir()
    msa_b.mkdir()
    (msa_a / "main.a3m").write_text(">query\nACDE\n", encoding="utf-8")
    (msa_b / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                    "main_msa_file_paths": [str(msa_a)],
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": ["B"],
                    "sequence": "BBBB",
                    "main_msa_file_paths": [str(msa_b)],
                },
            ]
        }
    )

    job = ScreeningJob(
        base_query=base_query,
        mutations=[],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    row = MutationScreeningRunner(backend=FakeBackend()).run(job)[0]

    assert row.sequence_cache_hits == 2


def test_mutation_screening_runner_reuses_only_invariant_chain_for_multiple_mutations(
    tmp_path,
):
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
        mutations=[
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G"),
            MutationSpec(chain_id="A", position_1based=3, from_residue="D", to_residue="Y"),
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    rows = MutationScreeningRunner(backend=FakeBackend()).run(job)

    assert [row.sequence_cache_hits for row in rows] == [1, 1]


def test_mutation_screening_runner_places_wt_first_when_requested(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    rows = MutationScreeningRunner(backend=FakeBackend()).run(job)

    assert [row.mutation_id for row in rows] == ["WT", "A_C2G"]


def test_mutation_screening_runner_places_wt_last_when_requested(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=False,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    rows = MutationScreeningRunner(backend=FakeBackend()).run(job)

    assert [row.mutation_id for row in rows] == ["A_C2G", "WT"]


def test_mutation_screening_runner_sets_cache_hit_from_invariant_chain_reuse(tmp_path):
    msa_dir = tmp_path / "alignments_B"
    msa_dir.mkdir()
    (msa_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"},
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
        mutations=[
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    row = MutationScreeningRunner(backend=FakeBackend()).run(job)[0]

    assert row.cache_hit is True
    assert row.query_result_cache_hit is False


def test_mutation_screening_runner_sets_cache_hit_false_without_any_reuse(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    row = MutationScreeningRunner(backend=FakeBackend()).run(job)[0]

    assert row.cache_hit is False
    assert row.sequence_cache_hits == 0


def test_mutation_screening_runner_writes_manifest_and_csv_consistently(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    rows = MutationScreeningRunner(backend=FakeBackend()).run(job)
    manifest = json.loads((job.output_dir / "screening_manifest.json").read_text())

    with (job.output_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert manifest["rows_written"] == len(rows) == len(csv_rows)
    assert manifest["num_mutations"] == 1
    assert manifest["include_wt"] is True
    assert csv_rows[0]["mutation_id"] == "WT"
    assert csv_rows[1]["mutation_id"] == "A_C2G"


def test_mutation_screening_runner_persists_query_result_cache_files(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )

    rows = MutationScreeningRunner(backend=FakeBackend()).run(job)
    cache_files = list((job.cache_dir / "results").glob("*.json"))

    assert len(cache_files) == len(rows)


def test_mutation_screening_runner_marks_second_run_rows_as_cached(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
    )
    backend = FakeBackend()
    runner = MutationScreeningRunner(backend=backend)

    runner.run(job)
    second_rows = runner.run(job)

    assert backend.calls == 2
    assert all(row.cache_hit is True for row in second_rows)
    assert all(row.query_result_cache_hit is True for row in second_rows)


def test_mutation_screening_runner_consumes_prepared_jobs_as_they_finish(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G"),
            MutationSpec(chain_id="A", position_1based=3, from_residue="D", to_residue="Y"),
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        num_cpu_workers=2,
        max_inflight_queries=2,
    )

    runner = MutationScreeningRunner(backend=FakeBackend())
    original_prepare = runner._prepare_single_job
    call_order = []

    def delayed_prepare(*args, **kwargs):
        mutation_spec = args[1]
        assert mutation_spec is not None
        if mutation_spec.position_1based == 2:
            time.sleep(0.05)
        prepared = original_prepare(*args, **kwargs)
        assert isinstance(prepared, PreparedMutationJob)
        call_order.append(prepared.query_id)
        return prepared

    runner._prepare_single_job = delayed_prepare

    rows = runner.run(job)

    assert call_order[0] == "screen_A_D3Y"
    assert [row.query_id for row in rows] == ["screen_A_D3Y", "screen_A_C2G"]


def test_mutation_screening_runner_can_disable_query_result_cache(tmp_path):
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
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G")
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=True,
        run_baseline_first=True,
        output_policy="metrics_only",
        cache_query_results=False,
        num_cpu_workers=1,
        max_inflight_queries=1,
    )
    backend = FakeBackend()
    runner = MutationScreeningRunner(backend=backend)

    first_rows = runner.run(job)
    second_rows = runner.run(job)
    manifest = json.loads((job.output_dir / "screening_manifest.json").read_text())

    assert len(first_rows) == len(second_rows) == 2
    assert backend.calls == 4
    assert not (job.cache_dir / "results").exists()
    assert all(row.query_result_cache_hit is False for row in second_rows)
    assert manifest["cache_query_results"] is False


def test_mutation_screening_runner_batches_prepared_jobs_when_configured(tmp_path):
    base_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDEFG"},
            ]
        }
    )

    job = ScreeningJob(
        base_query=base_query,
        mutations=[
            MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G"),
            MutationSpec(chain_id="A", position_1based=3, from_residue="D", to_residue="Y"),
            MutationSpec(chain_id="A", position_1based=4, from_residue="E", to_residue="K"),
        ],
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        subprocess_batch_size=2,
        num_cpu_workers=3,
        max_inflight_queries=3,
    )
    backend = BatchRecordingBackend()

    rows = MutationScreeningRunner(backend=backend).run(job)
    manifest = json.loads((job.output_dir / "screening_manifest.json").read_text())

    assert len(rows) == 3
    assert backend.run_calls == 0
    assert sum(backend.batch_sizes) == 3
    assert max(backend.batch_sizes) == 2
    assert any(batch_size > 1 for batch_size in backend.batch_sizes)
    assert manifest["subprocess_batch_size"] == 2


def test_mutation_screening_runner_dispatches_partial_batch_after_timeout(tmp_path):
    base_query = Query.model_validate(
        {
            "chains": [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDEFG"},
            ]
        }
    )

    mutations = [
        MutationSpec(chain_id="A", position_1based=2, from_residue="C", to_residue="G"),
        MutationSpec(chain_id="A", position_1based=3, from_residue="D", to_residue="Y"),
        MutationSpec(chain_id="A", position_1based=4, from_residue="E", to_residue="K"),
    ]
    job = ScreeningJob(
        base_query=base_query,
        mutations=mutations,
        output_dir=tmp_path / "screening",
        cache_dir=tmp_path / "cache",
        include_wt=False,
        output_policy="metrics_only",
        subprocess_batch_size=3,
        dispatch_partial_batches=True,
        batch_gather_timeout_seconds=0.01,
        num_cpu_workers=1,
        max_inflight_queries=1,
    )
    backend = BatchRecordingBackend()
    runner = SlowPrepareRunner(
        delays_by_mutation_id={
            mutations[0].mutation_id: 0.0,
            mutations[1].mutation_id: 0.05,
            mutations[2].mutation_id: 0.05,
        },
        backend=backend,
    )

    rows = runner.run(job)
    manifest = json.loads((job.output_dir / "screening_manifest.json").read_text())

    assert len(rows) == 3
    assert backend.run_calls == 0
    assert backend.batch_sizes[0] == 1
    assert manifest["dispatch_partial_batches"] is True
    assert manifest["batch_gather_timeout_seconds"] == 0.01
