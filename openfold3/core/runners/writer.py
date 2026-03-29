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

"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


def _batch_bool_flags(value) -> list[bool]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [bool(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [bool(v) for v in value]
    return [bool(value)]


class NumpyEncoder(json.JSONEncoder):
    r"""Custom JSON encoder for handling numpy data types.

    https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _take_batch_dim(x, b: int):
    if isinstance(x, torch.Tensor):
        if len(x.shape) > 1:
            return x[b].cpu().float().numpy()
        else:
            return x
    if isinstance(x, dict):
        return {k: _take_batch_dim(v, b) for k, v in x.items()}
    return x


def _take_sample_dim(x, s: int):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.shape[0] == 1:
            return x[0]
        return x[s]
    if isinstance(x, dict):
        return {k: _take_sample_dim(v, s) for k, v in x.items()}
    return x


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(
        self,
        output_dir: Path,
        pae_enabled: bool = False,
        structure_format: str = "pdb",
        full_confidence_output_format: str = "json",
        write_features: bool = False,
        write_latent_outputs: bool = False,
        metrics_only: bool = False,
        cif_only: bool = False,
        summary_filename: str = "summary.jsonl",
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.pae_enabled = pae_enabled
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_output_format
        self.write_features = write_features
        self.write_latent_outputs = write_latent_outputs
        self.metrics_only = metrics_only
        self.cif_only = cif_only
        self.summary_path = self.output_dir / summary_filename

        # Track successfully predicted samples
        self.success_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.failed_queries = []

    @staticmethod
    def _to_jsonable(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: OF3OutputWriter._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OF3OutputWriter._to_jsonable(v) for v in value]
        return value

    def append_summary_row(self, row: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(self._to_jsonable(row), cls=NumpyEncoder) + "\n")

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
        make_ost_compatible: bool = True,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(
            atom_array,
            output_file,
            include_bonds=True,
            make_ost_compatible=make_ost_compatible,
        )

    def get_pae_confidence_scores(self, confidence_scores, atom_array):
        pae_confidence_scores = {}
        single_value_keys = [
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
        ]

        for key in single_value_keys:
            pae_confidence_scores[key] = confidence_scores[key]

        # Get map from asym id to chain id
        renum_ids = np.unique(atom_array.chain_id, return_inverse=True)[1] + 1
        asym_id_to_chain_id = {
            k: v
            for (k, v) in set(
                [
                    (int(x[0]), str(x[1]))
                    for x in zip(renum_ids, atom_array.chain_id, strict=True)
                ]
            )
        }

        # Asym id -> chain id for chain_ptm
        pae_confidence_scores["chain_ptm"] = {
            asym_id_to_chain_id[int(k)]: v
            for k, v in confidence_scores["chain_ptm"].items()
        }

        # Asym id -> chain id for chain_pair_iptm
        pae_confidence_scores["chain_pair_iptm"] = {}
        for k, v in confidence_scores["chain_pair_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["chain_pair_iptm"][f"({k1}, {k2})"] = v

        # Asym id -> chain id for bespoke_iptm
        pae_confidence_scores["bespoke_iptm"] = {}
        for k, v in confidence_scores["bespoke_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["bespoke_iptm"][f"({k1}, {k2})"] = v
        return pae_confidence_scores

    def write_confidence_scores(
        self,
        confidence_scores: dict[str, np.ndarray],
        atom_array: structure.AtomArray,
        output_prefix: Path,
    ):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["pde"]
        gpde = confidence_scores["gpde"]
        aggregated_confidence_scores = {"avg_plddt": np.mean(plddt), "gpde": gpde}

        if self.pae_enabled:
            logger.info("Recording PAE confidence outputs")
            aggregated_confidence_scores |= self.get_pae_confidence_scores(
                confidence_scores, atom_array
            )

        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        out_file_agg.write_text(
            json.dumps(aggregated_confidence_scores, indent=4, cls=NumpyEncoder)
        )

        # Full confidence scores
        full_confidence_scores = {"plddt": plddt, "pde": pde}
        out_fmt = self.full_confidence_format
        out_file_full = Path(f"{output_prefix}_confidences.{out_fmt}")

        if not self.metrics_only:
            if out_fmt == "json":
                out_file_full.write_text(
                    json.dumps(
                        full_confidence_scores,
                        indent=4,
                        cls=NumpyEncoder,
                    )
                )
            elif out_fmt == "npz":
                np.savez_compressed(out_file_full, **full_confidence_scores)

        return self._to_jsonable(aggregated_confidence_scores)

    def write_all_outputs(self, batch: dict, outputs: dict, confidence_scores: dict):
        """Writes all outputs for a given batch."""

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]

            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            # Extract attributes for the current batch
            atom_array_batch = batch["atom_array"][b]
            predicted_coords_batch = (
                outputs["atom_positions_predicted"][b].cpu().float().numpy()
            )
            confidence_scores_batch = (
                None if self.cif_only else _take_batch_dim(confidence_scores, b)
            )

            # Iterate over all diffusion samples
            for s in range(sample_size):
                file_prefix = output_subdir / f"{query_id}_seed_{seed}_sample_{s + 1}"
                file_prefix.parent.mkdir(parents=True, exist_ok=True)

                predicted_coords_sample = predicted_coords_batch[s]
                if self.cif_only:
                    confidence_scores_sample = None
                    plddt_values = np.zeros(len(atom_array_batch), dtype=np.float32)
                else:
                    confidence_scores_sample = _take_sample_dim(
                        confidence_scores_batch, s
                    )
                    plddt_values = confidence_scores_sample["plddt"]

                structure_file = Path(f"{file_prefix}_model.{self.structure_format}")
                if not self.metrics_only:
                    # Save predicted structure
                    self.write_structure_prediction(
                        atom_array=atom_array_batch,
                        predicted_coords=predicted_coords_sample,
                        plddt=plddt_values,
                        output_file=structure_file,
                    )

                aggregated_scores: dict[str, Any] = {}
                aggregated_confidence_path: str | None = None
                if not self.cif_only:
                    aggregated_scores = self.write_confidence_scores(
                        confidence_scores=confidence_scores_sample,
                        output_prefix=file_prefix,
                        atom_array=atom_array_batch,
                    )
                    aggregated_confidence_path = str(
                        Path(f"{file_prefix}_confidences_aggregated.json")
                    )
                self.append_summary_row(
                    {
                        "query_id": query_id,
                        "mutation_id": query_id,
                        "seed": int(seed),
                        "sample_index": s + 1,
                        "aggregated_confidence_path": aggregated_confidence_path,
                        "derived_interface_metrics": {},
                        **aggregated_scores,
                    }
                )

            def fetch_cur_batch(t):
                # Get tensor for current batch dim
                # Remove expanded sample dim if it exists to get original tensor shapes
                if t.ndim < 2:
                    return t

                cur_feats = t[b : b + 1].squeeze(1)  # noqa: B023
                return cur_feats.detach().clone().cpu()

            file_prefix = output_subdir / f"{query_id}_seed_{seed}"

            # Write out input feature dictionary
            if self.write_features and not self.metrics_only:
                out_file = Path(f"{file_prefix}_batch.pt")
                cur_batch = tensor_tree_map(fetch_cur_batch, batch, strict_type=False)
                torch.save(cur_batch, out_file)
                del cur_batch

            # Write out latent reps / raw model outputs
            if self.write_latent_outputs and not self.metrics_only:
                out_file = Path(f"{file_prefix}_latent_output.pt")
                cur_output = tensor_tree_map(
                    fetch_cur_batch, outputs, strict_type=False
                )
                torch.save(cur_output, out_file)
                del cur_output

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Skip repeated samples
        repeated_flags = _batch_bool_flags(batch.get("repeated_sample"))
        if repeated_flags and all(repeated_flags):
            return
        if repeated_flags and any(repeated_flags):
            raise ValueError(
                "Mixed repeated/non-repeated samples in an output-writing batch are "
                "not supported. Re-run with data_module_args.batch_size=1 for this input."
            )

        self.total_count += 1

        # Skip and track failed samples
        if outputs is None:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            return

        batch, outputs = outputs
        confidence_scores = outputs.get("confidence_scores")

        # Write predictions and confidence scores
        # Optionally write out input features and latent outputs
        try:
            write_started = time.perf_counter()
            self.write_all_outputs(
                batch=batch,
                outputs=outputs,
                confidence_scores=confidence_scores,
            )
            write_seconds = time.perf_counter() - write_started
            logger.info(
                "Writer timings for query_id(s) %s: batch_size=%s write=%.2fs",
                ", ".join(batch["query_id"]),
                len(batch["query_id"]),
                write_seconds,
            )
            self.success_count += 1
        except Exception as e:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            logger.exception(
                f"Failed to write predictions for query_id(s) "
                f"{', '.join(batch['query_id'])}: {e}"
            )

        del batch, outputs

    def on_predict_end(self, trainer, pl_module):
        """
        Print summary of inference run. Includes a timeout failsafe
        for distributed runs.
        """

        try:
            # Gather summary data from all processes
            final_summary_data = {
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "failed_queries": self.failed_queries,
            }

            gathered_data = [final_summary_data]
            if dist.is_available() and dist.is_initialized():
                gathered_data = [None] * trainer.world_size
                dist.all_gather_object(gathered_data, final_summary_data)

            if trainer.is_global_zero:
                # Aggregate the results from all processes on Rank 0
                total_queries = sum(data["total"] for data in gathered_data)
                success_count = sum(data["success"] for data in gathered_data)
                failed_count = sum(data["failed"] for data in gathered_data)
                final_failed_list = [
                    item for data in gathered_data for item in data["failed_queries"]
                ]

                self._write_summary(
                    total_queries=total_queries,
                    success_count=success_count,
                    failed_count=failed_count,
                    failed_list=final_failed_list,
                    global_rank=trainer.global_rank,
                    is_complete=True,
                )

        except RuntimeError as e:
            # TODO: Due to additional sync PL does outside of this callback,
            #  this won't be reached before the timeout error occurs.
            #  Leaving this here for now in case we refactor the prediction
            #  logic to avoid the extra syncs.
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logger.warning(
                    f"[Rank {trainer.global_rank}] Distributed sync timed out! "
                    f"Writing local results to a fallback log."
                )

                self._write_summary(
                    total_queries=self.total_count,
                    success_count=self.success_count,
                    failed_count=self.failed_count,
                    failed_list=self.failed_queries,
                    global_rank=trainer.global_rank,
                    is_complete=False,
                )

            else:
                # Re-raise unexpected runtime errors
                raise e

    def _write_summary(
        self,
        total_queries: int,
        success_count: int,
        failed_count: int,
        failed_list: list,
        global_rank: int = 0,
        is_complete=True,
    ):
        """Helper to format the final summary."""
        if is_complete:
            status = "COMPLETE"
            out_file = self.output_dir / "summary.txt"
        else:
            status = f"INCOMPLETE (Rank {global_rank})"
            out_file = self.output_dir / f"fallback_summary_rank_{global_rank}.txt"

        summary = [
            "\n" + "=" * 50,
            f"    PREDICTION SUMMARY ({status})    ",
            "=" * 50,
            f"Total Queries Processed: {total_queries}",
            f"  - Successful Queries:  {success_count}",
            f"  - Failed Queries:      {failed_count}",
        ]

        if failed_list:
            failed_str = ", ".join(sorted(list(set(failed_list))))
            summary.append(f"\nFailed Queries: {failed_str}")

        summary.append("=" * 50 + "\n")
        summary = "\n".join(summary)

        out_file.write_text(summary)

        if is_complete:
            print(summary)
        else:
            logger.warning(
                f"Fallback summary for Rank {global_rank} saved to: {out_file}"
            )
