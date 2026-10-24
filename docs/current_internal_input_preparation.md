# Current internal input preparation

This stage prepares structure and sequence inputs for future `current_internal` recalculation jobs. It does not run FoldX, Rosetta, OpenFold, ESM2, or article metrics.

## Why this stage exists

FoldX and Rosetta jobs need a clean chain structure where the mutation site exists and the wild-type residue matches the normalized mutation table. ESM2 and OpenFold-related jobs need sequence inputs. Running methods before these checks would mix method failures with input-quality failures.

## Main commands

Preview planned PDB IDs:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --dry-run
```

Run the full preparation:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --all --workers 8 --force
```

`--force` is required when derived outputs already exist. Without `--force`, the script refuses to overwrite manifests, FASTA files, jobs-with-inputs, or QC reports.

## Partial modes

Download only:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --download-structures
```

This mode downloads raw mmCIF files and updates download logs only. It does not update structure manifests, sequence manifests, residue mappings, jobs-with-inputs, or QC reports. This prevents a download-only run from replacing valid readiness data with incomplete validation results.

Prepare structures:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --prepare-structures --force
```

This mode requires raw structures, creates or revalidates chain PDB files, validates mutation sites, and writes structure manifest, structure errors, and residue mapping.

Prepare sequences:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --prepare-sequences --force
```

This mode reads existing prepared chain PDB files and writes FASTA files plus the sequence manifest. It does not rebuild structure readiness or jobs-with-inputs.

Full mode:

```bash
python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --all --workers 8 --force
```

This runs download, structure preparation, sequence preparation, jobs-with-inputs update, and QC report generation.

## Outputs

Raw downloaded mmCIF files:

```text
data/structures/raw/current_internal/
```

Chain-only PDB files:

```text
data/structures/chain/current_internal/
```

FASTA files:

```text
data/sequences/current_internal/current_internal_pdb_chain_sequences.fasta
data/sequences/current_internal/current_internal_openfold_inputs.fasta
data/sequences/current_internal/current_internal_esm2_inputs.fasta
```

Manifests:

```text
data/manifests/current_internal_structure_manifest.csv
data/manifests/current_internal_sequence_manifest.csv
data/manifests/current_internal_structure_errors.csv
data/manifests/current_internal_sequence_errors.csv
data/manifests/current_internal_residue_mapping.csv
```

Updated job table:

```text
data/processed/current_internal/current_internal_recalc_jobs_with_inputs.csv
```

## Reading manifests

`current_internal_structure_manifest.csv` is one row per normalized mutation. It records raw and chain structure paths, download/preparation status, mutation-site presence, structure WT residue, WT match status, and structure quality flags.

`current_internal_sequence_manifest.csv` is one row per unique PDB chain sequence, not one row per mutation. A single chain sequence can cover many mutations.

`input_ready` in `current_internal_recalc_jobs_with_inputs.csv` means the job has the inputs needed for the next execution stage. For FoldX/Rosetta experimental jobs, this requires a prepared chain structure and a WT residue match. For ESM2, this requires a sequence. OpenFold-structure jobs remain `pending_openfold_structure` until predicted structures are generated.

## Existing chain validation

If a chain PDB already exists and `--force` is not used, the script revalidates that existing chain PDB instead of trusting the raw mmCIF. If the chain file cannot be parsed or the mutation-site residue does not match the normalized wild-type residue, experimental FoldX/Rosetta jobs remain blocked and the structure errors include `chain_file_validation_failed` or a WT mismatch flag.

## Handling WT mismatches

Rows flagged as `wt_residue_mismatch` must be manually audited before scoring. Do not automatically rewrite mutations or renumber structures at this stage.

## Next step

After this stage, use `current_internal_recalc_jobs_with_inputs.csv` to drive method-specific execution layers for FoldX, Rosetta, OpenFold structure generation, and ESM2 scoring.
