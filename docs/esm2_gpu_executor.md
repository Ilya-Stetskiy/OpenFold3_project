# ESM2 GPU executor

This layer prepares and optionally runs `esm2_gpu_optional` jobs for the `current_internal` dataset. It follows the shared execution contract and uses `current_internal_recalc_jobs_resolved.csv` as the source of truth.

By default the executor does not run ESM2. It supports dry-run planning, preflight checks, materialized job directories, a one-job smoke run, and explicit full execution.

## Configure

Create the local machine-specific config:

```bash
cp configs/tools/esm2.example.yaml configs/tools/esm2.yaml
nano configs/tools/esm2.yaml
```

`configs/tools/esm2.yaml` is gitignored. Check:

- `esm2.python_bin`
- `esm2.model_name`
- `esm2.device`
- `esm2.timeout_seconds`
- `paths.sequence_root`
- `paths.results_root`
- `paths.logs_root`

## Commands

Plan only:

```bash
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --dry-run
```

Validate config and inputs without running ESM2:

```bash
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --preflight
```

Materialize job directories and pending manifests without scoring:

```bash
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --materialize-plan --max-jobs 2 --force
```

Run one server smoke job:

```bash
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --smoke-run --max-jobs 1
```

Run the full selected batch explicitly:

```bash
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --run --workers 1 --resume
```

## Job Selection

The runner selects only:

- `method = esm2`
- `structure_source = sequence_only`
- `input_ready = true`

It does not use legacy results and does not run FoldX, Rosetta, or OpenFold.

## Outputs

Each job writes under:

```text
results/current_internal/esm2/sequence_only/{mutation_record_id}/
```

Expected files:

- `command.txt`
- `stdout.log`
- `stderr.log`
- `job_input.json`
- `parsed_result.json`
- `job_manifest.json`
- `esm2_work/`
- `raw/`

The stage manifest is:

```text
results/current_internal/_manifests/esm2_gpu_optional_jobs.csv
results/current_internal/_manifests/esm2_gpu_optional_jobs.json
```

The per-stage completion marker after real execution is:

```text
data/run_state/current_internal/gpu_heavy.esm2_gpu_optional.completed.json
```

The executor does not create the global `gpu_heavy.completed.json`; OpenFold stages may still be pending.

## Parsed Result

`parsed_result.json` uses:

- `dataset_name`
- `method = esm2`
- `structure_source = sequence_only`
- `mutation_record_id`
- `job_id`
- `mutation`
- `chain_id`
- `esm2_score`
- `esm2_score_unit = log_probability_difference`
- `model_name`
- `scoring_mode`
- `parser_status`
- `error_message`

The default scorer uses a masked-marginal approximation: log probability of mutant residue minus log probability of wild-type residue at the mutation position.
