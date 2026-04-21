from __future__ import annotations


def apply_mutation(sequence: str, position: int, wt: str, mut: str) -> str:
    """Apply a single 1-based substitution to an amino-acid sequence."""

    if position < 1:
        raise ValueError("position must be >= 1")
    if len(wt) != 1 or len(mut) != 1:
        raise ValueError("wt and mut must be single-letter amino acid codes")
    index = position - 1
    if index >= len(sequence):
        raise ValueError("position is outside of sequence bounds")
    observed = sequence[index]
    if observed != wt:
        raise ValueError(
            f"Wild-type residue mismatch at position {position}: expected {wt}, found {observed}"
        )
    return sequence[:index] + mut + sequence[index + 1 :]
