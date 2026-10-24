# Execution contract

This document defines the common result contract for future FoldX, Rosetta, ESM2, and OpenFold executors.

The current layer is infrastructure only. It does not run FoldX, Rosetta, OpenFold, ESM2, or evaluation metrics.

## Recalc jobs vs execution manifests

`data/processed/current_internal/current_internal_recalc_jobs_resolved.csv` is the source of truth for what should be executed after structure overrides. Future executors must read that table and filter only jobs that are ready for their stage.

Execution manifests are runtime records written under `results/`. They describe what a concrete executor attempted, which files it used, which outputs it produced, and how the job ended. They do not replace the resolved recalc jobs table.

## Standard result layout

Every method job should write to:

```text
results/{dataset}/{method}/{structure_source}/{mutation_record_id}/
```

Standard files:

- `command.txt`
- `stdout.log`
- `stderr.log`
- `raw/`
- `parsed_result.json`
- `job_manifest.json`

Stage-level manifests are written to:

```text
results/{dataset}/_manifests/{stage_name}_jobs.csv
results/{dataset}/_manifests/{stage_name}_jobs.json
```

## Job statuses

Allowed job statuses are defined in `configs/execution_contract.yaml`:

- `pending`
- `blocked`
- `skipped`
- `running`
- `success`
- `failed`
- `parse_failed`
- `timeout`
- `cancelled`

Terminal statuses are:

- `success`
- `failed`
- `skipped`
- `blocked`
- `parse_failed`
- `timeout`
- `cancelled`

## Stage statuses

Allowed stage statuses:

- `not_started`
- `running`
- `completed`
- `completed_with_failures`
- `failed`
- `cancelled`

## Completion markers

Completion markers live under:

```text
data/run_state/{dataset}/{stage_group}.completed.json
```

A stage-group completion marker may be written only after a real executor stage ran, a stage manifest exists, and all planned jobs have terminal statuses. Dry-runs must not create completion markers. Dry-runs may write:

```text
data/run_state/{dataset}/{stage_group}.dry_run_plan.json
```

`postprocess` waits for completed `cpu_heavy` and `gpu_heavy` markers because OpenFold outputs, FoldX/Rosetta outputs, parsed scores, and confidence files are produced on different paths and may finish at different times.

## Utilities

Future executors should use `scripts/run/execution_utils.py` for:

- YAML/CSV/JSON reads and writes;
- safe writes that refuse accidental overwrite without `--force`;
- job output directory construction;
- job and stage manifest writing;
- completion marker writing;
- stage-specific filtering of resolved jobs.

The dry-run skeleton in `scripts/run/run_jobs_skeleton.py` demonstrates the expected filtering and materialized-plan behavior without launching heavy tools.

## Legacy results

Legacy method outputs from `data/raw/current_internal/results.csv` are archived for regression/debug comparison only. They are not source of truth for article metrics. All new executors must use `current_internal_recalc_jobs_resolved.csv` and write new outputs under `results/current_internal/`.

