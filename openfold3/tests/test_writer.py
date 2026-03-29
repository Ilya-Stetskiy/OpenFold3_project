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

import numpy as np
import pytest  # noqa: F401  - used for pytest tmp fixture
import torch
from biotite import structure
from biotite.structure.io import pdb, pdbx

from openfold3.core.runners.writer import OF3OutputWriter


class TestPredictionWriter:
    @pytest.mark.parametrize(
        "structure_format",
        ["pdb", "cif"],
        ids=lambda x: x,
    )
    def test_written_coordinates(self, tmp_path, structure_format):
        atom1 = structure.Atom([1, 2, 3], chain_id="A", element="C")
        atom2 = structure.Atom([2, 3, 4], chain_id="A", element="N")
        atom3 = structure.Atom([3, 4, 5], chain_id="B", element="O")

        atom_array = structure.array([atom1, atom2, atom3])
        atom_array.entity_id = np.array(["A", "A", "B"])
        atom_array.molecule_type_id = np.array(["0", "0", "1"])
        atom_array.pdbx_formal_charge = np.array(["1", "1", "1"])

        # add extra dimension for sample
        new_coords = np.array(
            [
                [2.0, 2.0, 2.0],
                [3.5, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        dummy_plddt = np.array([0.9, 0.8, 0.7])

        output_writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=False,
            structure_format=structure_format,
            full_confidence_output_format="json",
        )
        tmp_file = tmp_path / f"TEST.{structure_format}"
        output_writer.write_structure_prediction(
            atom_array, new_coords, dummy_plddt, tmp_file, False
        )

        match structure_format:
            case "cif":
                read_file = pdbx.CIFFile.read(tmp_file)
                parsed_structure = pdbx.get_structure(read_file)

            case "pdb":
                parsed_structure = pdb.PDBFile.read(tmp_file).get_structure()

        parsed_coords = parsed_structure.coord[0]
        np.testing.assert_array_equal(parsed_coords, new_coords, strict=False)

    def _load_full_confidence_scores(self, output_file_path):
        output_fmt = output_file_path.suffix.lstrip(".")
        match output_fmt:
            case "json":
                actual_full_scores = json.loads(output_file_path.read_text())
                actual_full_scores = {
                    k: np.array(v) for k, v in actual_full_scores.items()
                }
            case "npz":
                actual_full_scores = np.load(output_file_path)
        return actual_full_scores

    @pytest.mark.parametrize(
        "output_fmt",
        ["json", "npz"],
        ids=lambda x: x,
    )
    def test_confidence_writer_without_pae(
        self, tmp_path, output_fmt, dummy_atom_array
    ):
        n_tokens = 3
        n_atoms = 5
        dummy_atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])

        confidence_scores = {
            "plddt": np.random.uniform(size=n_atoms),
            "pde_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pde": np.random.uniform(size=(n_tokens, n_tokens)),
            "gpde": np.float32(16.2),
        }

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=False,
            full_confidence_output_format=output_fmt,
        )
        output_prefix = tmp_path / "test"
        writer.write_confidence_scores(
            confidence_scores, dummy_atom_array, output_prefix
        )

        # Check aggregated confidence scores
        expected_agg_scores = {
            "avg_plddt": np.mean(confidence_scores["plddt"]),
            "gpde": confidence_scores["gpde"],
        }
        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        actual_agg_scores = json.loads(out_file_agg.read_text())
        assert expected_agg_scores == actual_agg_scores

        # Check full confidence scores:
        expected_full_scores = {
            "plddt": confidence_scores["plddt"],
            "pde": confidence_scores["pde"],
        }
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
        actual_full_scores = self._load_full_confidence_scores(out_file_full)

        for k in expected_full_scores:
            assert k in actual_full_scores, f"Key {k} not found in actual scores"
            np.testing.assert_array_equal(
                expected_full_scores[k], actual_full_scores[k]
            )

    @pytest.mark.parametrize(
        "output_fmt",
        ["json", "npz"],
        ids=lambda x: x,
    )
    def test_confidence_writer_with_pae(self, tmp_path, output_fmt, dummy_atom_array):
        n_tokens = 3
        n_atoms = 5
        dummy_atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])

        confidence_scores = {
            "plddt": np.random.uniform(size=n_atoms),
            "pde_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pde": np.random.uniform(size=(n_tokens, n_tokens)),
            "gpde": np.random.uniform(size=(1,)),
            "pae_probs": np.random.uniform(size=(n_tokens, n_tokens, 64)),
            "pae": np.random.uniform(size=(n_tokens, n_tokens)),
            "iptm": np.random.uniform(size=(1,)),
            "ptm": np.random.uniform(size=(1,)),
            "disorder": np.random.uniform(size=(1,)),
            "has_clash": np.float32(0.0),
            "sample_ranking_score": np.random.uniform(size=(1,)),
            "chain_ptm": {
                "1": np.random.uniform(size=(1,)),
                "2": np.random.uniform(size=(1,)),
            },
            "chain_pair_iptm": {
                "(1, 2)": np.random.uniform(size=(1,)),
            },
            "bespoke_iptm": {
                "(1, 2)": np.random.uniform(size=(1,)),
            },
        }

        output_writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=True,
            full_confidence_output_format=output_fmt,
        )

        output_prefix = tmp_path / "test"
        output_writer.write_confidence_scores(
            confidence_scores, dummy_atom_array, output_prefix
        )

        expected_agg_score_keys = [
            "avg_plddt",
            "gpde",
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
            "chain_ptm",
            "chain_pair_iptm",
            "bespoke_iptm",
        ]

        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        actual_agg_scores = json.loads(out_file_agg.read_text())
        assert set(expected_agg_score_keys) == set(actual_agg_scores.keys())

        # Check full confidence scores:
        expected_full_scores = {
            "plddt": confidence_scores["plddt"],
            "pde": confidence_scores["pde"],
        }
        out_file_full = Path(f"{output_prefix}_confidences.{output_fmt}")
        actual_full_scores = self._load_full_confidence_scores(out_file_full)

        for k in expected_full_scores:
            assert k in actual_full_scores, f"Key {k} not found in actual scores"
            np.testing.assert_array_equal(
                expected_full_scores[k], actual_full_scores[k]
            )

    def test_skips_none_output(self, tmp_path):
        class DummyMock:
            pass

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            structure_format="pdb",
            full_confidence_output_format="npz",
        )
        trainer = DummyMock()
        pl_module = DummyMock()

        writer.on_predict_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch={"query_id": "query_id"},
            batch_idx=0,
        )

        assert writer.failed_count == 1
        assert writer.success_count == 0

    def test_metrics_only_writer_skips_structure_and_full_confidence(self, tmp_path):
        atom_array = structure.array(
            [
                structure.Atom([1.0, 2.0, 3.0], chain_id="A"),
                structure.Atom([2.0, 3.0, 4.0], chain_id="A"),
            ]
        )
        atom_array.entity_id = np.array(["A", "A"])
        atom_array.molecule_type_id = np.array(["0", "0"])
        atom_array.pdbx_formal_charge = np.array(["0", "0"])

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=True,
            structure_format="pdb",
            full_confidence_output_format="json",
            metrics_only=True,
            summary_filename="summary.jsonl",
        )

        batch = {
            "atom_array": [atom_array],
            "seed": [42],
            "query_id": ["mut_A_A10G"],
        }
        outputs = {
            "atom_positions_predicted": torch.tensor(
                [[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]], dtype=torch.float32
            )
        }
        confidence_scores = {
            "plddt": np.array([[[0.9, 0.8]]], dtype=np.float32),
            "pde": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
            "gpde": np.array([np.float32(1.5)]),
            "iptm": np.array([np.float32(0.7)]),
            "ptm": np.array([np.float32(0.6)]),
            "disorder": np.array([np.float32(0.1)]),
            "has_clash": np.array([np.float32(0.0)]),
            "sample_ranking_score": np.array([np.float32(0.8)]),
            "chain_ptm": {"1": np.array([np.float32(0.6)])},
            "chain_pair_iptm": {"(1, 1)": np.array([np.float32(0.7)])},
            "bespoke_iptm": {"(1, 1)": np.array([np.float32(0.7)])},
        }

        writer.write_all_outputs(batch, outputs, confidence_scores)

        prefix = (
            tmp_path
            / "mut_A_A10G"
            / "seed_42"
            / "mut_A_A10G_seed_42_sample_1"
        )
        assert not Path(f"{prefix}_model.pdb").exists()
        assert not Path(f"{prefix}_confidences.json").exists()
        assert Path(f"{prefix}_confidences_aggregated.json").exists()

        summary_rows = (tmp_path / "summary.jsonl").read_text().splitlines()
        assert len(summary_rows) == 1
        summary = json.loads(summary_rows[0])
        assert summary["query_id"] == "mut_A_A10G"
        assert summary["mutation_id"] == "mut_A_A10G"
        assert summary["sample_index"] == 1

    def test_cif_only_writer_skips_confidence_outputs(self, tmp_path):
        atom_array = structure.array(
            [
                structure.Atom([1.0, 2.0, 3.0], chain_id="A"),
                structure.Atom([2.0, 3.0, 4.0], chain_id="A"),
            ]
        )
        atom_array.entity_id = np.array(["A", "A"])
        atom_array.molecule_type_id = np.array(["0", "0"])
        atom_array.pdbx_formal_charge = np.array(["0", "0"])

        writer = OF3OutputWriter(
            output_dir=tmp_path,
            pae_enabled=True,
            structure_format="cif",
            full_confidence_output_format="json",
            cif_only=True,
            summary_filename="summary.jsonl",
        )

        batch = {
            "atom_array": [atom_array],
            "seed": [42],
            "query_id": ["mut_A_A10G"],
        }
        outputs = {
            "atom_positions_predicted": torch.tensor(
                [[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]], dtype=torch.float32
            )
        }

        writer.write_all_outputs(batch, outputs, confidence_scores=None)

        prefix = (
            tmp_path
            / "mut_A_A10G"
            / "seed_42"
            / "mut_A_A10G_seed_42_sample_1"
        )
        assert Path(f"{prefix}_model.cif").exists()
        assert not Path(f"{prefix}_confidences.json").exists()
        assert not Path(f"{prefix}_confidences_aggregated.json").exists()

        summary_rows = (tmp_path / "summary.jsonl").read_text().splitlines()
        assert len(summary_rows) == 1
        summary = json.loads(summary_rows[0])
        assert summary["query_id"] == "mut_A_A10G"
        assert summary["aggregated_confidence_path"] is None
