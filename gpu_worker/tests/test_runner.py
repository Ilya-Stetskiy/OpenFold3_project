from __future__ import annotations

import json
from pathlib import Path

from gpu_worker.runner import OpenFoldWorkerRunner
from gpu_worker.schemas import LeaseJob, WorkerConfig


def _config(tmp_path: Path) -> WorkerConfig:
    repo = tmp_path / "openfold-3"
    repo.mkdir()
    return WorkerConfig(
        worker_id="worker-1",
        worker_token="token",
        server_url="http://server.test",
        openfold_project_dir=tmp_path,
        openfold_repo_dir=repo,
        results_dir=tmp_path / "results",
    )


def test_prepare_predict_batch_writes_query_json(tmp_path: Path) -> None:
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {
                "queries": [
                    {
                        "query_id": "q1",
                        "molecules": [
                            {
                                "type": "protein",
                                "id": "A",
                                "sequence": "acde",
                            }
                        ],
                    }
                ],
                "use_msa_server": False,
                "use_templates": False,
            },
            "upload": {"url": "http://server.test/upload"},
        }
    )

    command = OpenFoldWorkerRunner(_config(tmp_path)).prepare(lease, tmp_path / "work")
    payload = json.loads((tmp_path / "work" / "query.json").read_text(encoding="utf-8"))

    assert payload["queries"]["q1"]["chains"][0]["molecule_type"] == "protein"
    assert payload["queries"]["q1"]["chains"][0]["chain_ids"] == ["A"]
    assert payload["queries"]["q1"]["chains"][0]["sequence"] == "ACDE"
    assert "predict" in command.cmd
    assert "--use_msa_server" in command.cmd


def test_point_mutation_variant_uses_screen_mutations(tmp_path: Path) -> None:
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-2",
            "lease_id": "lease-2",
            "job_type": "variant_batch",
            "payload": {
                "base_query": {
                    "molecules": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                        }
                    ]
                },
                "query_prefix": "scan",
                "variants": [
                    {
                        "variant_id": "A_C2G",
                        "mutations": [
                            {
                                "chain_id": "A",
                                "position_1based": 2,
                                "to_residue": "G",
                            }
                        ],
                    }
                ],
            },
            "upload": {"url": "http://server.test/upload"},
        }
    )

    command = OpenFoldWorkerRunner(_config(tmp_path)).prepare(lease, tmp_path / "work")
    job = json.loads(
        (tmp_path / "work" / "screening_job.json").read_text(encoding="utf-8")
    )

    assert "screen-mutations" in command.cmd
    assert job["mutations"] == [
        {
            "chain_id": "A",
            "position_1based": 2,
            "from_residue": "C",
            "to_residue": "G",
        }
    ]
    assert job["cleanup_query_outputs"] is False


def test_arbitrary_variant_batch_falls_back_to_predict(tmp_path: Path) -> None:
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-3",
            "lease_id": "lease-3",
            "job_type": "variant_batch",
            "payload": {
                "base_query": {
                    "molecules": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                        }
                    ]
                },
                "query_prefix": "panel",
                "variants": [
                    {
                        "variant_id": "full_seq",
                        "molecules": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "AAAA",
                            }
                        ],
                    }
                ],
            },
            "upload": {"url": "http://server.test/upload"},
        }
    )

    command = OpenFoldWorkerRunner(_config(tmp_path)).prepare(lease, tmp_path / "work")
    payload = json.loads((tmp_path / "work" / "query.json").read_text(encoding="utf-8"))

    assert "predict" in command.cmd
    assert payload["queries"]["panel__full_seq"]["chains"][0]["sequence"] == "AAAA"
