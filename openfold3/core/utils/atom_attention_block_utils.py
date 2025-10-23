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

import math

import torch


def get_query_block_padding(n_atom: int, n_query: int) -> int:
    """
    Calculate padding for a structure with n_atoms so that the sequence
    is evenly divisible by n_query.

    Args:
        n_atom:
            Number of atoms
        n_query:
            Number of queries (block height)

    Returns:
        pad_len_right_q: Padding for the query seqlen dim
    """
    return (n_query - n_atom % n_query) % n_query


def get_block_indices(
    n_atom: int, n_query: int, n_key: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the indices to gather such that the block centers
    match the subset centers in Alg. 7

    Args:
        n_atom:
            Number of atoms
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        device:
            Device to create the tensors on

    Returns:
        key_block_idxs:
            [N_blocks, N_key] Indices to gather for keys
        invalid_mask:
            [N_blocks, N_key] Boolean mask for invalid indices
    """
    offset = n_query // 2
    num_blocks = math.ceil(n_atom / n_query)

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query

    initial_gathers = (
        subset_centers[:, None]
        + torch.arange(-n_key // 2, n_key // 2, device=device)[None, :]
    )

    initial_gathers = initial_gathers.int()

    # For normal cases, shift windows to be fully in-bounds.
    # For each row, calculate how much its start index is below 0.
    underflow = torch.relu(-initial_gathers[:, 0])

    # For each row, calculate how much its end index is above the max valid index.
    overflow = torch.relu(initial_gathers[:, -1] - (n_atom - 1))

    # The total shift required for each row.
    # We prioritize correcting underflow: if a window underflows, shift it
    # right to start at 0. Only if it does not underflow, do we check
    # for overflow and shift it left.
    total_shift = torch.where(underflow > 0, underflow, -overflow)

    # Apply the calculated shift to each row.
    # We add `[:, None]` to the shift tensor to broadcast the per-row shift
    # value across all columns of that row in `initial_gathers`.
    final_gathers = initial_gathers + total_shift[:, None]

    # Create a boolean mask to identify all indices that are invalid.
    invalid_mask = (final_gathers < 0) | (final_gathers >= n_atom)

    # Create "safe" indices by clamping the generated ones to the valid range.
    # This prevents torch.gather from throwing an error.
    safe_indices = torch.clamp(final_gathers, 0, n_atom - 1)

    return safe_indices.flatten(), invalid_mask.flatten()


def get_pair_atom_block_mask(
    atom_mask: torch.Tensor,
    num_blocks: int,
    n_query: int,
    n_key: int,
    pad_len_right_q: int,
    key_block_idxs: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Generates the q/k block pair mask from the atom mask.

    Args:
        atom_mask:
            [*, N_atom] Atom mask
        num_blocks:
            Number of blocks
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        pad_len_right_q:
            Right padding for the query dimension
        key_block_idxs:
            [N_blocks * N_key] 1D flat tensor of key indices
        invalid_mask:
            [N_blocks * N_key] 1D flat boolean mask for invalid indices

    Returns:
        atom_pair_mask:
            [*, N_blocks, N_query, N_key]
    """
    # [*, N_atom] -> [*, N_blocks, N_query]
    atom_mask_q = torch.nn.functional.pad(atom_mask, (0, pad_len_right_q), value=0.0)
    atom_mask_q = atom_mask_q.reshape((*atom_mask.shape[:-1], num_blocks, n_query))

    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    flat_batch_size = int(math.prod(batch_dims))

    atom_mask_flat = atom_mask.reshape(flat_batch_size, n_atom)

    # Create flat index/mask, shape [flat_batch_size, num_blocks * n_key]
    index_flat = key_block_idxs.view(1, -1).expand(flat_batch_size, -1)
    mask_flat = invalid_mask.view(1, -1).expand(flat_batch_size, -1)

    # Gather on flattened tensors
    atom_mask_k_flat = torch.gather(atom_mask_flat, 1, index_flat.long())
    atom_mask_k_flat.masked_fill_(mask_flat, 0)

    # Reshape back to original batch dims
    atom_mask_k = atom_mask_k_flat.reshape((*batch_dims, num_blocks, n_key))

    # [*, N_blocks, N_query, N_key]
    atom_pair_mask = atom_mask_q[..., None] * atom_mask_k[..., None, :]

    return atom_pair_mask


def convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Convert single atom representation to q/k blocks for attention.
    Optionally convert the atom mask to a 2D mask to account for padding.

    Args:
        ql:
            [*, N_atom, c_atom] Atom single representation
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        atom_mask:
            [*, N_atom] Mask for token or atom-level embedding (Optional)

    Returns:
        ql_query:
            [*, N_blocks, N_query, c_atom] Atom single representation
        ql_key:
            [*, N_blocks, N_key, c_atom] Atom single representation
        mask:
            [*, N_blocks, N_query, N_key] 2D mask for atom-level embedding
    """
    batch_dims = ql.shape[:-2]
    n_atom, n_dim = ql.shape[-2:]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q = get_query_block_padding(n_atom=n_atom, n_query=n_query)

    # Pad and convert ql to blocks of width n_query
    # [*, N_atom, c_atom] -> [*, N_blocks, N_query, c_atom]
    ql_query = torch.nn.functional.pad(ql, (0, 0, 0, pad_len_right_q), value=0.0)
    ql_query = ql_query.reshape((*batch_dims, num_blocks, n_query, n_dim))

    key_block_idxs, invalid_mask = get_block_indices(
        n_atom=n_atom, n_query=n_query, n_key=n_key, device=ql.device
    )

    flat_batch_size = int(math.prod(batch_dims))
    ql_flat = ql.reshape(flat_batch_size, n_atom, n_dim)

    # Create flat index/mask, shape [flat_batch_size, num_blocks * n_key, n_dim]
    index_flat = key_block_idxs.view(1, -1, 1).expand(flat_batch_size, -1, n_dim)
    mask_flat = invalid_mask.view(1, -1, 1).expand(flat_batch_size, -1, n_dim)

    # Gather on flattened tensors
    ql_key_flat = torch.gather(ql_flat, 1, index_flat.long())
    ql_key_flat.masked_fill_(mask_flat, 0)

    # Reshape back to original batch dims
    ql_key = ql_key_flat.reshape((*batch_dims, num_blocks, n_key, n_dim))

    atom_pair_mask = None
    if atom_mask is not None:
        atom_pair_mask = get_pair_atom_block_mask(
            atom_mask=atom_mask,
            num_blocks=num_blocks,
            n_query=n_query,
            n_key=n_key,
            pad_len_right_q=pad_len_right_q,
            key_block_idxs=key_block_idxs,  # Pass 1D flat tensor
            invalid_mask=invalid_mask,  # Pass 1D flat tensor
        )

    return ql_query, ql_key, atom_pair_mask


def convert_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair atom representation to blocks for attention.

    Args:
        batch:
            Feature dictionary
        zij_trunk:
            [*, N_token, N_token, c_atom_pair] Pair trunk embedding
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)

    Returns:
        plm:
            [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
    """
    # Get atom_to_token_index to map each token to the corresponding
    # number of atoms for broadcasting
    atom_to_token_index = batch["atom_to_token_index"]

    batch_dims = zij_trunk.shape[:-3]
    n_atom = atom_to_token_index.shape[-1]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q = get_query_block_padding(n_atom=n_atom, n_query=n_query)

    # Pad and convert atom_to_token_index to blocks of width n_query
    atom_to_token_index_q = torch.nn.functional.pad(
        atom_to_token_index, (0, pad_len_right_q), value=0.0
    )

    # [*, N_atom] -> [*, N_blocks, N_query]
    atom_to_token_index_q = atom_to_token_index_q.reshape(
        (*batch_dims, num_blocks, n_query)
    )

    key_block_idxs, invalid_mask = get_block_indices(
        n_atom=n_atom, n_query=n_query, n_key=n_key, device=zij_trunk.device
    )

    key_block_idxs_flat_1d = key_block_idxs
    invalid_mask_flat_1d = invalid_mask

    # [1, num_blocks * n_key]
    key_block_idxs = key_block_idxs.reshape((1, -1))

    # Reshape invalid_mask for flat_batch_size.
    # [1, num_blocks, n_key]
    invalid_mask = invalid_mask.reshape(1, num_blocks, n_key)

    flat_batch_size = int(math.prod(batch_dims))

    # Flatten batch dims
    zij_trunk = zij_trunk.reshape(flat_batch_size, *zij_trunk.shape[-3:])
    q_indices = atom_to_token_index_q.reshape(
        flat_batch_size, num_blocks, n_query
    ).long()

    # Flatten atom_to_token_index
    atom_to_token_index_flat = atom_to_token_index.reshape(flat_batch_size, n_atom)

    # Create flat index, shape [flat_batch_size, num_blocks * n_key]
    key_block_idxs_flat = key_block_idxs.view(1, -1).expand(flat_batch_size, -1)

    # Gather on flattened tensor
    k_indices_flat = torch.gather(
        atom_to_token_index_flat, 1, key_block_idxs_flat.long()
    )

    # Reshape to final k_indices shape
    k_indices = k_indices_flat.reshape(flat_batch_size, num_blocks, n_key)

    # Create batch index for the flattened dim
    batch_index = torch.arange(
        flat_batch_size, device=zij_trunk.device, dtype=torch.long
    ).view(-1, 1, 1, 1)

    # [*, N_blocks, N_query, N_key, c_atom_pair]
    plm = zij_trunk[batch_index, q_indices.unsqueeze(-1), k_indices.unsqueeze(-2)]

    plm.masked_fill_(
        invalid_mask[..., None, :, None].expand(
            (flat_batch_size, num_blocks, n_query, n_key, plm.shape[-1])
        ),
        0,
    )

    # Compute atom pair mask for masking out padding
    # The broadcast op above will set the token at index 0 for all padding,
    # we need to reset this
    atom_mask = batch["atom_mask"]

    atom_pair_mask = get_pair_atom_block_mask(
        atom_mask=atom_mask,
        num_blocks=num_blocks,
        n_query=n_query,
        n_key=n_key,
        pad_len_right_q=pad_len_right_q,
        key_block_idxs=key_block_idxs_flat_1d,  # Pass 1D flat tensor
        invalid_mask=invalid_mask_flat_1d,  # Pass 1D flat tensor
    )

    # Reshape plm back to original batch dimensions before returning
    plm = plm.reshape((*batch_dims, num_blocks, n_query, n_key, plm.shape[-1]))

    # Mask out padding
    plm = plm * atom_pair_mask.unsqueeze(-1)

    return plm
