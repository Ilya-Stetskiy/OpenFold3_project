from __future__ import annotations

from .canonical import MUTATION_RE


def apply_mutation(sequence: str, mutation: str) -> str:
    match = MUTATION_RE.match(mutation.strip().upper())
    if match is None:
        raise ValueError(f"Invalid mutation format: {mutation}")
    wt, position_text, mut = match.groups()
    position = int(position_text)
    if position < 1 or position > len(sequence):
        raise ValueError(
            f"Mutation position {position} is outside sequence length {len(sequence)}"
        )
    observed = sequence[position - 1].upper()
    if observed != wt:
        raise ValueError(
            f"Wild-type residue mismatch at {position}: expected {wt}, found {observed}"
        )
    return sequence[: position - 1] + mut + sequence[position :]
