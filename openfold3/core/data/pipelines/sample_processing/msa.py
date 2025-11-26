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

"""This module contains SampleProcessingPipelines for MSA features."""

from functools import partial

from openfold3.core.config.config_utils import DirectoryPathOrNone
from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInput,
    MsaSampleProcessorInputInference,
    MsaSampleProcessorInputTrain,
)
from openfold3.core.data.io.sequence.msa import (
    MsaSampleParser,
    MsaSampleParserInference,
    MsaSampleParserTrain,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    create_main,
    create_paired,
    create_paired_from_preprocessed,
    create_query_seqs,
    find_monomer_homomer,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings


class MsaSampleProcessor:
    """Base class for MSA sample processing."""

    def __init__(self, config: MSASettings):
        self.config = config
        self.msa_sample_parser = MsaSampleParser(config=config)
        self.query_seq_processor = create_query_seqs
        self.paired_msa_processor = partial(
            create_paired,
            max_rows_paired=config.max_rows_paired,
            min_chains_paired_partial=config.min_chains_paired_partial,
            pairing_mask_keys=config.pairing_mask_keys,
            max_seq_per_species=config.max_seq_per_species,
            msas_to_pair=config.msas_to_pair,
        )
        self.main_msa_processor = partial(
            create_main,
            max_rows=config.max_rows,
            aln_order=config.aln_order,
            subsample_main=config.subsample_main,
            keep_subsampled_order=config.keep_subsampled_order,
        )

    def create_query_seq(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_main_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
        chain_id_to_paired_msa: dict[str, MsaArray] | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def __call__(self, input: MsaSampleProcessorInput) -> MsaArrayCollection:
        # Parse MSAs
        msa_array_collection = self.msa_sample_parser(input=input)

        # Create dicts with the processed query, paired and main MSA data per chain
        chain_id_to_query_seq = self.create_query_seq(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_paired_msa = self.create_paired_msa(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_main_msa, chain_id_to_profile, chain_id_to_deletion_mean = (
            self.create_main_msa(
                input=input,
                msa_array_collection=msa_array_collection,
                chain_id_to_paired_msa=chain_id_to_paired_msa,
            )
        )

        # Update MsaArrayCollection with processed MSA data
        msa_array_collection.set_state_processed(
            chain_id_to_query_seq=chain_id_to_query_seq,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
            chain_id_to_main_msa=chain_id_to_main_msa,
            chain_id_to_profile=chain_id_to_profile,
            chain_id_to_deletion_mean=chain_id_to_deletion_mean,
        )

        return msa_array_collection


# TODO: test
class MsaSampleProcessorTrain(MsaSampleProcessor):
    """Pipeline for MSA sample processing for training."""

    def __init__(
        self,
        config: MSASettings,
        *,
        alignment_array_directory: DirectoryPathOrNone = None,
        alignment_db_directory: DirectoryPathOrNone = None,
        alignment_index: dict | None = None,
        alignments_directory: DirectoryPathOrNone = None,
        use_roda_monomer_format: bool = False,
    ):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserTrain(
            config=config,
            alignment_array_directory=alignment_array_directory,
            alignment_db_directory=alignment_db_directory,
            alignment_index=alignment_index,
            alignments_directory=alignments_directory,
            use_roda_monomer_format=use_roda_monomer_format,
        )

    def create_query_seq(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Determine whether to do pairing
            if not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    def create_main_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Create main MSA arrays
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = (
                self.main_msa_processor(
                    msa_array_collection=msa_array_collection,
                    chain_id_to_paired_msa=chain_id_to_paired_msa,
                )
            )
        else:
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = {}, {}, {}
        return chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean


class MsaSampleProcessorInference(MsaSampleProcessor):
    """Pipeline for MSA sample processing for inference."""

    def __init__(self, config: MSASettings):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserInference(config=config)

    def create_query_seq(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if (len(msa_array_collection.rep_id_to_query_seq) > 0) & input.use_msas:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_paired_msas
        ):
            # Use precomputed paired MSAs
            # TODO modularize better
            if len(msa_array_collection.rep_id_to_paired_msa) > 0:
                chain_id_to_paired_msa = create_paired_from_preprocessed(
                    msa_array_collection=msa_array_collection,
                    max_rows_paired=self.config.max_rows_paired,
                    paired_msa_order=self.config.paired_msa_order,
                )
            # Pair online from main MSAs
            elif not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    def create_main_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_main_msas
        ):
            # Create main MSA arrays
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = (
                self.main_msa_processor(
                    msa_array_collection=msa_array_collection,
                    chain_id_to_paired_msa=chain_id_to_paired_msa,
                )
            )
        else:
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = {}, {}, {}
        return chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean
