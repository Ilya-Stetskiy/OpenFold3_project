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

import pandas as pd

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_of3 import (
    BaseOF3Dataset,
)

logger = logging.getLogger(__name__)


@register_dataset
class MonomerDataset(BaseOF3Dataset):
    def __init__(self, dataset_config: dict) -> None:
        """Initializes a ProteinMonomerDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # Datapoint cache
        self.create_datapoint_cache()

        # Dataset configuration
        self.apply_crop = True
        self.crop = dataset_config.crop.model_dump()

    def create_datapoint_cache(self):
        """Creates the datapoint_cache for uniform sampling.

        Creates a Dataframe storing a flat list of structure_data keys and sets
        corresponding datapoint probabilities all to 1. Used for mapping FROM the
        dataset_cache in the SamplerDataset and TO the dataset_cache in the
        getitem.
        """
        # TODO: rename PDB ID to MGnify ID or more generic name
        sample_ids = list(self.dataset_cache.structure_data.keys())
        sample_indices = list(
            [
                entry_data.chains["1"].index
                for entry_data in self.dataset_cache.structure_data.values()
            ]
        )
        datapoint_cache_unsorted = pd.DataFrame(
            {
                "pdb_id": sample_ids,
                "index": sample_indices,
                "datapoint_probabilities": [1.0] * len(sample_ids),
            }
        )
        self.datapoint_cache = datapoint_cache_unsorted.sort_values("index")[
            ["pdb_id", "datapoint_probabilities"]
        ]
