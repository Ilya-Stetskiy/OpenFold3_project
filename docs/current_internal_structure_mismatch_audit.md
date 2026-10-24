# Current internal structure mismatch audit

This audit diagnoses normalized mutations whose expected wild-type residue does not cleanly agree with the prepared experimental PDB chain structure.

It does not modify raw data, `current_internal_mutations.csv`, `current_internal_recalc_jobs_with_inputs.csv`, or any structure files. It also does not run FoldX, Rosetta, OpenFold, or ESM2.

## Why WT residue mismatches matter

FoldX and Rosetta experimental-structure jobs assume that the structure contains the wild-type residue at the mutation position. If the structure residue differs, the mutation may be incorrectly defined, the residue numbering may be shifted, the PDB chain may already contain a variant residue, or the dataset annotation may not match the structure.

Running scoring jobs before resolving this can produce misleading results.

## Possible causes

- `numbering_shift_candidate`: the expected wild-type amino acid appears nearby, suggesting numbering mismatch.
- `same_mutant_already_in_structure`: the structure residue already equals the mutant amino acid.
- `insertion_code_candidate`: nearby residues share the same residue number with insertion codes.
- `dataset_annotation_problem`: expected wild-type amino acid is not found nearby.
- `pdb_variant_candidate`: the PDB structure may represent a variant sequence rather than the stated wild type.

## Outputs

```text
data/manifests/current_internal_structure_mismatch_audit.csv
data/manifests/current_internal_structure_mismatch_audit.json
reports/current_internal/current_internal_structure_mismatch_audit.md
reports/current_internal/current_internal_structure_mismatch_audit.json
```

## Recommended actions

`candidate_position_remap` means the wild-type amino acid was found exactly once within +/-10 residues and may indicate a numbering shift. This is still only a candidate.

`manual_review` means there is ambiguous local evidence, such as multiple nearby matches or insertion-code ambiguity.

`possible_pdb_already_mutant` means the structure residue equals the mutant amino acid.

`exclude_from_experimental_structure_jobs` means no obvious safe correction was found; keep FoldX/Rosetta experimental jobs blocked unless a manual audit resolves it.

`keep_blocked` means the diagnostic evidence is insufficient.

## Manual decision process

Start from the CSV fields `nearby_positions`, `nearby_sequence_window`, `candidate_corrected_position`, and `affected_jobs`. Compare the PDB residue numbering, mutation annotation, and source dataset row before deciding whether to remap, exclude, or keep blocked.
