# Copyright 2025 AlQuraishi Laboratory
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

# %%
import logging
import random
import traceback

import torch

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    check_invalid_feature_dict,
)
from openfold3.core.data.framework.single_datasets.monomer import (
    MonomerDataset,
)

logger = logging.getLogger(__name__)


@register_dataset
class RNAMonomerDataset(MonomerDataset):
    def __init__(self, dataset_config: dict) -> None:
        """Initializes a RNAMonomerDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # All samples are RNA
        self.single_moltype = "RNA"

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset.

        Note: The data pipeline is modularized at the getitem level to enable
        subclassing for profiling without code duplication. See
        logging_datasets.py for an example."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]

        # TODO: Remove debug logic
        if not self.debug_mode:
            sample_data = self.create_all_features(
                pdb_id=datapoint["pdb_id"],
                preferred_chain_or_interface=None,
                return_atom_arrays=False,
                return_crop_strategy=False,
            )

            features = sample_data["features"]
            features["pdb_id"] = pdb_id
            features["preferred_chain_or_interface"] = "none"
            return features
        else:
            try:
                sample_data = self.create_all_features(
                    pdb_id=datapoint["pdb_id"],
                    preferred_chain_or_interface=None,
                    return_atom_arrays=False,
                    return_crop_strategy=False,
                )

                features = sample_data["features"]

                features["pdb_id"] = pdb_id
                features["preferred_chain_or_interface"] = "none"

                check_invalid_feature_dict(features)

                return features

            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(
                    "-" * 40
                    + "\n"
                    + f"Failed to process RNAMonomerDataset entry {pdb_id}:"
                    + f" {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)
