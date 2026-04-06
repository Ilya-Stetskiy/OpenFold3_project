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

from pathlib import Path

import pytest
import torch

from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom


class DummyPredictRunner:
    def __init__(self):
        self.global_rank = 0
        self.global_step = 0
        self.log_dir = Path(".")
        self.seed_calls = []

    def reseed(self, seed):
        self.seed_calls.append(seed)

    def __call__(self, batch):
        batch_size = len(batch["query_id"])
        outputs = {"atom_positions_predicted": torch.zeros((batch_size, 1, 1, 3))}
        return batch, outputs

    def _compute_confidence_scores(self, batch, outputs):
        return {"dummy": torch.tensor(1.0)}

    def _log_predict_exception(self, e, query_id):
        raise AssertionError(f"Unexpected predict exception for {query_id}: {e}")


def test_predict_step_accepts_batched_queries_with_shared_seed():
    runner = DummyPredictRunner()
    batch = {
        "query_id": ["q1", "q2"],
        "seed": torch.tensor([7, 7]),
        "repeated_sample": torch.tensor([False, False]),
        "valid_sample": torch.tensor([True, True]),
    }

    returned_batch, outputs = OpenFold3AllAtom.predict_step(runner, batch, batch_idx=0)

    assert returned_batch["seed"] == [7, 7]
    assert runner.seed_calls == [7]
    assert "confidence_scores" in outputs


def test_predict_step_rejects_mixed_seeds_in_batched_queries():
    runner = DummyPredictRunner()
    batch = {
        "query_id": ["q1", "q2"],
        "seed": torch.tensor([7, 8]),
        "repeated_sample": torch.tensor([False, False]),
        "valid_sample": torch.tensor([True, True]),
    }

    with pytest.raises(ValueError, match="identical seeds"):
        OpenFold3AllAtom.predict_step(runner, batch, batch_idx=0)


def test_writer_accepts_batched_repeated_sample_tensor_with_none_outputs(tmp_path):
    writer = OF3OutputWriter(
        output_dir=tmp_path,
        structure_format="cif",
        full_confidence_output_format="json",
    )

    writer.on_predict_batch_end(
        trainer=object(),
        pl_module=object(),
        outputs=None,
        batch={
            "query_id": ["q1", "q2"],
            "repeated_sample": torch.tensor([False, False]),
        },
        batch_idx=0,
    )

    assert writer.failed_count == 1
