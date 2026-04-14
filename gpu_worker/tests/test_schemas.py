from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpu_worker.schemas import LeaseJob, PredictBatchPayload, VariantBatchPayload


def test_predict_batch_payload_accepts_high_level_queries() -> None:
    payload = PredictBatchPayload.model_validate(
        {
            "queries": [
                {
                    "query_id": "ubq",
                    "molecules": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A"],
                            "sequence": "ACDE",
                        }
                    ],
                }
            ],
            "num_diffusion_samples": 1,
            "num_model_seeds": 1,
        }
    )

    assert payload.queries[0].query_id == "ubq"
    assert payload.num_diffusion_samples == 1


def test_variant_batch_payload_supports_point_mutations() -> None:
    payload = VariantBatchPayload.model_validate(
        {
            "base_query": {
                "molecules": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": "ACDE",
                    }
                ]
            },
            "variants": [
                {
                    "variant_id": "A_C2G",
                    "mutations": [
                        {
                            "chain_id": "A",
                            "position_1based": 2,
                            "from_residue": "C",
                            "to_residue": "G",
                        }
                    ],
                }
            ],
        }
    )

    assert payload.variants[0].mutations[0].to_residue == "G"


def test_lease_job_parses_server_contract() -> None:
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {
                "queries": [
                    {
                        "molecules": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                ]
            },
            "limits": {"max_runtime_seconds": 10},
            "upload": {"url": "http://example.test/upload", "method": "PUT"},
        }
    )

    assert lease.limits.max_runtime_seconds == 10
    assert lease.upload.url == "http://example.test/upload"


def test_lease_job_requires_upload_target() -> None:
    with pytest.raises(ValidationError):
        LeaseJob.model_validate(
            {
                "job_id": "job-1",
                "lease_id": "lease-1",
                "job_type": "predict_batch",
                "payload": {"queries": []},
            }
        )
