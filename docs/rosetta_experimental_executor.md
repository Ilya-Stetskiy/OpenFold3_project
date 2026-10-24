# Rosetta experimental executor

This layer prepares, validates, and can explicitly run Rosetta jobs for `current_internal` experimental chain structures.

The default behavior is safe: without `--run` or `--smoke-run`, Rosetta is not executed.

## Configure Rosetta

Copy the example config and edit the binary path on the server:

```bash
cp configs/tools/rosetta.example.yaml configs/tools/rosetta.yaml
nano configs/tools/rosetta.yaml
```

`configs/tools/rosetta.yaml` is local and gitignored. Set:

```yaml
rosetta:
  binary_path: /real/server/path/to/rosetta/ddg_monomer
```

## Job selection

The executor reads only:

```text
data/processed/current_internal/current_internal_recalc_jobs_resolved.csv
```

It selects:

- `method = rosetta`
- `structure_source = experimental`
- `input_ready = true`

Blocked jobs, OpenFold-dependent jobs, and legacy results are not used as source of truth.

## Dry run and preflight

```bash
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --dry-run
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --preflight
```

Preflight checks:

- config path;
- Rosetta binary path and executable bit;
- resolved jobs CSV;
- per-job chain ID;
- per-job experimental structure path;
- mutation format;
- existing successful outputs when `--force` is not used.

Reports:

```text
reports/current_internal/rosetta_experimental_preflight.md
reports/current_internal/rosetta_experimental_preflight.json
```

## Materialize plan

```bash
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --materialize-plan --max-jobs 2 --force
```

This creates placeholder job directories and manifests only:

```text
results/current_internal/rosetta/experimental/{mutation_record_id}/
```

Files:

- `command.txt`
- `mutation_file.txt`
- `input_structure.pdb`
- `raw/`
- `rosetta_work/`
- `job_manifest.json`

It does not create `parsed_result.json` and does not run Rosetta.

## Smoke run and full run

On the server, after configuring the real binary:

```bash
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --smoke-run --max-jobs 1
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --run --workers 4 --resume
```

Real execution runs each job in its isolated `rosetta_work/` directory. Existing successful outputs are skipped unless `--force` is passed. Previous failed, parse-failed, timeout, or interrupted/running manifests are also skipped unless `--force` is passed.

## Parsed result

`parsed_result.json` uses:

- `dataset_name`
- `method`
- `structure_source`
- `mutation_record_id`
- `job_id`
- `mutation`
- `chain_id`
- `rosetta_ddg`
- `rosetta_unit`
- `number_of_runs`
- `raw_output_files`
- `parser_status`
- `error_message`

`rosetta_unit` is `Rosetta Energy Units`.

## Stage manifest and marker

Stage manifests:

```text
results/current_internal/_manifests/rosetta_experimental_jobs.csv
results/current_internal/_manifests/rosetta_experimental_jobs.json
```

Real `--run` writes only a per-stage marker:

```text
data/run_state/current_internal/cpu_heavy.rosetta_experimental.completed.json
```

It does not create global `cpu_heavy.completed.json`, because FoldX and optional ESM2 CPU stages may still be pending.

