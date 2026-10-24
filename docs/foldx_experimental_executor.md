# FoldX executor

This layer prepares and validates FoldX jobs for `current_internal`. It defaults to experimental chain structures and can also target OpenFold-derived structures with `--structure-source openfold`. It does not run the full FoldX batch unless `--run` is explicitly passed.

## Configure FoldX

Copy the example config and edit the binary path on the server:

```bash
cp configs/tools/foldx.example.yaml configs/tools/foldx.yaml
nano configs/tools/foldx.yaml
```

`configs/tools/foldx.yaml` is local and gitignored. Set:

```yaml
foldx:
  binary_path: /real/server/path/to/foldx
```

## Job selection

For experimental structures, the executor reads:

```text
data/processed/current_internal/current_internal_recalc_jobs_resolved.csv
```

It selects:

- `method = foldx`
- `structure_source = experimental`
- `input_ready = true`

For OpenFold structures, use:

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --dry-run
```

This reads the derived table:

```text
data/processed/current_internal/current_internal_recalc_jobs_openfold_ready.csv
```

and selects:

- `method = foldx`
- `structure_source = openfold`
- `input_ready = true`

Blocked jobs and legacy results are not used as source of truth.

Important: FoldX normally expects PDB input. OpenFold WT outputs are normalized as both `model.cif` and `model.pdb`; the derived OpenFold-ready jobs table should point FoldX to `model.pdb`.

## Dry run

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --dry-run
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --dry-run
```

This prints the selected job count and does not write result directories or run FoldX.

## Preflight

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --preflight
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --preflight
```

Preflight checks:

- config path;
- FoldX binary path and executable bit;
- resolved jobs CSV;
- per-job chain ID;
- per-job structure path for the selected source;
- mutation format;
- existing successful outputs when `--force` is not used.

Reports:

```text
reports/current_internal/foldx_experimental_preflight.md
reports/current_internal/foldx_experimental_preflight.json
reports/current_internal/foldx_openfold_preflight.md
reports/current_internal/foldx_openfold_preflight.json
```

If `configs/tools/foldx.yaml` is missing, preflight uses `configs/tools/foldx.example.yaml` for planning and reports the missing local config as a warning. Real execution still requires a valid executable binary.

## Materialize plan

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --materialize-plan --max-jobs 2 --force
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --materialize-plan --max-jobs 2 --force
```

This creates placeholder job directories and manifests only:

```text
results/current_internal/foldx/experimental/{mutation_record_id}/
results/current_internal/foldx/openfold/{mutation_record_id}/
```

Files:

- `command.txt`
- `individual_list.txt`
- `input_structure.pdb`
- `raw/`
- `foldx_work/`
- `job_manifest.json`

It does not create `parsed_result.json` and does not run FoldX.

## Smoke run

On the server, after configuring the real binary:

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --smoke-run --max-jobs 1
```

Smoke-run is intentionally limited. Use it to validate FoldX command behavior before a batch run.

## Full run

Explicit full execution:

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --run --workers 4 --resume
```

Real execution runs each job in its isolated `foldx_work/` directory. The executor copies `input_structure.pdb` and `individual_list.txt` into that directory, runs `RepairPDB`, then runs `BuildModel`.

Existing outputs are conservative:

- a successful `parsed_result.json` plus `job_manifest.json` is skipped unless `--force` is passed;
- previous failed, parse-failed, timeout, or interrupted/running manifests are skipped unless `--force` is passed;
- `--force` intentionally overwrites/retries selected jobs.

Use `--resume` for normal server continuation. The executor still refuses to overwrite successful outputs without `--force`.

## Parsed result

`parsed_result.json` uses:

- `dataset_name`
- `method`
- `structure_source`
- `mutation_record_id`
- `job_id`
- `mutation`
- `chain_id`
- `foldx_ddg`
- `foldx_unit`
- `number_of_runs`
- `raw_output_files`
- `parser_status`
- `error_message`

`foldx_unit` is `kcal/mol`.

## Stage manifest and marker

Stage manifests:

```text
results/current_internal/_manifests/foldx_experimental_jobs.csv
results/current_internal/_manifests/foldx_experimental_jobs.json
```

Real `--run` writes only a per-stage marker:

```text
data/run_state/current_internal/cpu_heavy.foldx_experimental.completed.json
```

It does not create global `cpu_heavy.completed.json`, because Rosetta and optional ESM2 CPU stages may still be pending.
