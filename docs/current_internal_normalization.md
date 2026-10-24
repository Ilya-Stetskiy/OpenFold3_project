# Current internal normalization

`current_internal` is the local pilot/internal DDG dataset. It is useful for pipeline development and regression checks, but it is not an independent benchmark.

The raw file is expected at:

```text
data/raw/current_internal/results.csv
```

This file may contain experimental DDG values and old FoldX/Rosetta/ESM2/OpenFold outputs. The old method outputs are archived as legacy results only. They are not final article results and must not be used as the source of truth for new metrics.

## Run

Preview planned outputs:

```bash
python3 scripts/data/normalize_current_internal.py --dataset current_internal --dry-run
```

Create or overwrite normalization outputs:

```bash
python3 scripts/data/normalize_current_internal.py --dataset current_internal --force
```

Limit future recalc jobs:

```bash
python3 scripts/data/normalize_current_internal.py --dataset current_internal --methods foldx rosetta esm2 --structure-sources experimental openfold sequence_only --force
```

## Outputs

Interim files:

```text
data/interim/current_internal/current_internal_raw_parsed.csv
data/interim/current_internal/current_internal_legacy_rows.csv
data/interim/current_internal/current_internal_normalization_issues.csv
data/interim/current_internal/current_internal_legacy_failures.csv
```

Processed files:

```text
data/processed/current_internal/current_internal_mutations.csv
data/processed/current_internal/current_internal_recalc_jobs.csv
data/processed/current_internal/current_internal_legacy_results.csv
```

Reports:

```text
reports/current_internal/current_internal_normalization_qc.md
reports/current_internal/current_internal_normalization_qc.json
```

## How to read the outputs

`current_internal_mutations.csv` has one row per unique single-site mutation. It contains parsed amino acids, position, experimental DDG, source row IDs, quality flags, and legacy-output availability.

If `pdb_id` and `chain_id` are not separate raw columns, the normalizer can parse compound `protein_id` values like `O74035_1IO2_A_D105A` into `uniprot_id`, `pdb_id`, `chain_id`, and mutation. If the explicit mutation column disagrees with the mutation encoded in `protein_id`, the row is flagged as `protein_id_mutation_mismatch`.

Experimental DDG units are normalized to `kcal/mol`. The raw `unit` column is treated as legacy method-output metadata and is written only as `legacy_unit`.

`current_internal_recalc_jobs.csv` has one pending job per mutation/method/structure-source combination. By default it creates FoldX and Rosetta jobs for `experimental` and `openfold` structures, plus ESM2 jobs for `sequence_only`.

`current_internal_legacy_results.csv` stores old method outputs with `legacy_*` columns. Use it only for regression/debug comparison after new results are generated.

## Benchmark policy

`current_internal` can be used for development and pilot checks. It must not be reported as an independent external benchmark. Final external validation belongs to `independent_benchmark`.
