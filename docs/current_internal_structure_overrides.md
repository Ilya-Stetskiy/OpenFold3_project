# Current internal structure overrides

This layer records manual decisions for current_internal structure mismatch cases.

It does not modify raw data, `current_internal_mutations.csv`, `current_internal_recalc_jobs_with_inputs.csv`, structures, or legacy outputs. It creates derived resolved files that the future execution layer can use.

## Why this layer exists

Some normalized mutations do not match the residue found in the prepared experimental PDB chain. Running FoldX or Rosetta directly on those jobs would make the mutation definition ambiguous.

The mismatch audit proposes possible reasons, but those proposals are not final corrections. The override layer turns each case into an explicit, editable curator decision.

## Files

Input:

```text
data/manifests/current_internal_structure_mismatch_audit.csv
data/manifests/current_internal_structure_manifest.csv
data/processed/current_internal/current_internal_recalc_jobs_with_inputs.csv
```

Editable decision file:

```text
data/manifests/current_internal_structure_override_template.csv
```

Derived outputs:

```text
data/manifests/current_internal_structure_manifest_resolved.csv
data/processed/current_internal/current_internal_recalc_jobs_resolved.csv
reports/current_internal/current_internal_structure_override_qc.md
reports/current_internal/current_internal_structure_override_qc.json
```

## Create the template

```bash
python3 scripts/data/create_current_internal_structure_override_template.py --dataset current_internal --force
```

The template is safe to edit manually. Re-running without `--force` will not overwrite it.

## Curator decisions

Allowed `curator_decision` values:

- `accept_original`: allow experimental FoldX/Rosetta jobs using the original mutation position.
- `accept_position_remap`: use `curated_position` after validating that residue_mapping contains the expected wild-type residue.
- `exclude`: keep experimental FoldX/Rosetta jobs blocked.
- `needs_review`: keep experimental FoldX/Rosetta jobs blocked until a curator decides.
- `accept_custom_structure`: use `curated_structure_path` if the file exists and residue validation passes.
- `accept_custom_chain`: use `curated_chain_id` if the prepared chain and residue validation pass.

Default decisions are conservative:

- candidate remaps start as `needs_review`;
- manual-review cases start as `needs_review`;
- possible already-mutant PDB cases start as `exclude`.

## Apply decisions

```bash
python3 scripts/data/apply_current_internal_structure_overrides.py --dataset current_internal --dry-run
python3 scripts/data/apply_current_internal_structure_overrides.py --dataset current_internal --force
```

Only `foldx + experimental` and `rosetta + experimental` readiness can change. OpenFold-related jobs remain pending OpenFold structures. ESM2 sequence-only jobs are not changed by structure overrides.

## Reading resolved outputs

`current_internal_structure_manifest_resolved.csv` contains the original position, resolved position, original and resolved WT residue validation, the curator decision, and the resolution status.

`current_internal_recalc_jobs_resolved.csv` keeps all original jobs and adds:

- `resolved_position`
- `structure_resolution_status`
- `structure_resolution_flags`
- `override_applied`

The future execution layer should read the resolved jobs table, not the original jobs-with-inputs table, after curation begins.

## Manual workflow

1. Open `data/manifests/current_internal_structure_override_template.csv`.
2. Review `recommended_action_from_audit`, `candidate_corrected_position`, local residue context in the audit CSV, and source annotations.
3. Set `curator_decision`.
4. Fill `decision_reason`, `curator`, and `decision_date`.
5. Run the apply script.
6. Inspect `reports/current_internal/current_internal_structure_override_qc.md`.

Do not edit raw results or normalized mutation tables to represent these decisions.
