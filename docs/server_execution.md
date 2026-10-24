# Server execution

This document describes the server run profile for the `current_internal` DDG pipeline.

The stage-group orchestrator is dry-run only. Concrete executors can run individual method stages only when their method-specific `--run` or `--smoke-run` flags are used.

Concrete executors must implement the shared result contract described in `docs/execution_contract.md`. That contract defines job statuses, stage statuses, result directory layout, job manifests, stage manifests, and completion marker rules.

The first concrete executor layers are documented in `docs/foldx_experimental_executor.md`, `docs/rosetta_experimental_executor.md`, `docs/esm2_gpu_executor.md`, and `docs/openfold_wt_executor.md`. Adaptive CPU-heavy planning is documented in `docs/cpu_heavy_adaptive_scheduler.md`. Method executors support dry-run, preflight, materialized plans, and selected real-run modes as documented per executor.

## Server stage model

1. Preprepare
2. CPU Heavy
3. GPU Heavy
4. Post-GPU CPU Heavy
5. Analysis Reporting

`preprepare` runs first. After it is complete, `cpu_heavy` and `gpu_heavy` can run independently and in parallel. `post_gpu_cpu_heavy` waits for both heavy stages because it runs CPU work that depends on GPU-generated OpenFold structures. `analysis_reporting` waits for post-GPU CPU outputs and only reads completed outputs to build statistics, metrics summaries, plots, and final reports.

```text
preprepare
    ├── cpu_heavy
    └── gpu_heavy
          ↓
post_gpu_cpu_heavy waits for both cpu_heavy and gpu_heavy
          ↓
analysis_reporting reads completed outputs only
```

## Stage groups

### Preprepare

Purpose: validate inputs, manifests, environment, and bundle contents before heavy execution.

Stages:

- `validate_dataset_registry`
- `validate_current_internal_inputs`
- `validate_resolved_jobs`
- `validate_structure_manifests`
- `build_evaluation_tasks`
- `build_server_bundles`
- `check_server_environment`

### CPU Heavy

Purpose: CPU-heavy execution that can run in parallel with GPU-heavy work.

Stages:

- `adaptive_cpu_heavy_scheduler`
- `foldx_experimental`
- `rosetta_experimental`
- `esm2_cpu_optional`
- `parse_cpu_outputs`

FoldX/Rosetta on experimental structures belong here. `adaptive_cpu_heavy_scheduler` computes conservative worker counts from available CPU/RAM/load/disk resources and passes those counts to method executors only when explicitly requested.

### GPU Heavy

Purpose: GPU-heavy OpenFold/optional ESM2 work.

Stages:

- `prepare_openfold_inputs`
- `openfold_wt_generation`
- `esm2_gpu_optional`
- `parse_gpu_outputs`

OpenFold WT generation belongs here.

OpenFold WT generation has dry-run, preflight, materialization, smoke-run, and explicit full-run modes.

### Post-GPU CPU Heavy

Purpose: CPU work that becomes possible only after GPU-generated OpenFold WT structures exist.

Stages:

- `collect_openfold_outputs`
- `update_openfold_readiness`
- `foldx_openfold`
- `rosetta_openfold`
- `parse_post_gpu_cpu_outputs`

FoldX/Rosetta on OpenFold structures belong here because they depend on GPU-generated OpenFold outputs. This stage writes derived readiness tables only; it does not rewrite raw data, normalized mutations, or `current_internal_recalc_jobs_resolved.csv`.

### Analysis Reporting

Purpose: statistics, metric summaries, plots, and final reports. This stage should be independent from method execution and must not run FoldX, Rosetta, OpenFold, or ESM2.

Stages:

- `run_ddg_evaluation`
- `run_global_structure_metrics`
- `run_local_mutation_structure_metrics`
- `run_confidence_error_metrics`
- `build_final_reports`

RMSD, TM-score, lDDT, GDT_TS, local contact precision, pLDDT summaries, pLDDT-vs-error analyses, tables, and plots belong here.

## Environment checks

```bash
python3 scripts/run/check_server_environment.py --stage-group all --server-config configs/server_paths.yaml
python3 scripts/run/check_server_environment.py --stage-group preprepare --server-config configs/server_paths.yaml
python3 scripts/run/check_server_environment.py --stage-group cpu_heavy --server-config configs/server_paths.yaml
python3 scripts/run/check_server_environment.py --stage-group gpu_heavy --server-config configs/server_paths.yaml
python3 scripts/run/check_server_environment.py --stage-group post_gpu_cpu_heavy --server-config configs/server_paths.yaml
python3 scripts/run/check_server_environment.py --stage-group analysis_reporting --server-config configs/server_paths.yaml
```

Reports are written to:

```text
reports/server_environment/
```

Tool paths are configured in a local file:

```text
configs/server_paths.yaml
```

Use `configs/server_paths.example.yaml` as the template. The local `server_paths.yaml` is ignored by git.

## Server paths config

`configs/server_paths.yaml` stores machine-specific server paths and tool locations. It is ignored by git because paths can contain local usernames, mount points, environment names, and private binary locations. For server execution checks, this file is the source of truth for server paths. The committable template is:

```text
configs/server_paths.example.yaml
```

Create the local/server config with:

```bash
cp configs/server_paths.example.yaml configs/server_paths.yaml
nano configs/server_paths.yaml
```

Fields to check before execution:

- `server.project_root`
- `server.data_root`
- `server.results_root`
- `server.logs_root`
- `server.reports_root`, if used
- `tools.foldx_binary`
- `tools.rosetta_bin_dir`
- `tools.openfold_env`
- `tools.python_bin`
- `gpu.cuda_visible_devices`
- `gpu.max_parallel_openfold_jobs`

Validate through CLI:

```bash
python3 scripts/run/check_server_environment.py --stage-group all --server-config configs/server_paths.yaml
```

The environment report shows the exact effective paths checked:

- `effective_project_root`
- `effective_data_root`
- `effective_results_root`
- `effective_logs_root`
- `effective_reports_root`

Writable checks use those effective paths from `configs/server_paths.yaml`. If `server.reports_root` is not set, the fallback is `project_root/reports` and the fallback is recorded in the JSON report.

Validate through notebook:

```text
notebooks/server_run_control_current_internal.ipynb
```

The notebook searches upward from the current Jupyter directory to find the repository root, using `configs/datasets.yaml` as the main marker. It then changes to that root, resolves relative paths from there, reads `configs/server_paths.yaml` when present, and falls back to `configs/server_paths.example.yaml` for read-only validation. It does not create or overwrite `configs/server_paths.yaml` unless the explicit safety flags are enabled.

## Dry-run orchestration

```bash
python3 scripts/run/run_stage_group.py --stage-group preprepare --dataset current_internal --dry-run --server-config configs/server_paths.yaml
python3 scripts/run/run_stage_group.py --stage-group cpu_heavy --dataset current_internal --dry-run --server-config configs/server_paths.yaml
python3 scripts/run/run_stage_group.py --stage-group gpu_heavy --dataset current_internal --dry-run --server-config configs/server_paths.yaml
python3 scripts/run/run_stage_group.py --stage-group post_gpu_cpu_heavy --dataset current_internal --dry-run --server-config configs/server_paths.yaml
python3 scripts/run/run_stage_group.py --stage-group analysis_reporting --dataset current_internal --dry-run --server-config configs/server_paths.yaml
```

The orchestrator reads completion markers from:

```text
data/run_state/current_internal/
```

Marker names:

```text
preprepare.completed.json
cpu_heavy.completed.json
gpu_heavy.completed.json
post_gpu_cpu_heavy.completed.json
analysis_reporting.completed.json
```

At this stage, the scripts only read marker status and print dependency warnings. They do not create heavy-stage completion markers.

Dry-run executor skeletons may write `*.dry_run_plan.json`, but they must not write `*.completed.json`. A completed marker is valid only after a real executor stage ran, a stage manifest exists, and all planned jobs have terminal statuses.

## Execution contract preview

Inspect resolved jobs:

```bash
python3 scripts/run/inspect_resolved_jobs.py --dataset current_internal
```

Preview future executor jobs without running heavy tools:

```bash
python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name foldx_experimental --dry-run
python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name rosetta_experimental --dry-run
python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name esm2_cpu_optional --dry-run
```

Adaptive CPU-heavy planning:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --preflight
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --run --resume --no-progress-bar
```

FoldX experimental has its own executor entrypoint:

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --dry-run
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --preflight
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --materialize-plan --max-jobs 2 --force
```

The same FoldX runner is reused for OpenFold-derived structures by changing the source:

```bash
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --dry-run
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --structure-source openfold --preflight
```

Real `--structure-source openfold --run` uses the FoldX-ready `model.pdb` generated during OpenFold output normalization. The original `model.cif` is kept for structural metrics and traceability.

After the first OpenFold server smoke-run, verify real CIF-to-PDB conversion before launching downstream FoldX/Rosetta-on-OpenFold jobs:

```bash
python3 scripts/executors/collect_openfold_outputs.py --dataset current_internal --dry-run
```

The OpenFold output is not considered ready unless `model.cif`, `model.pdb`, and `confidence.json` are all present.

Server smoke-run and full run require a real `configs/tools/foldx.yaml`:

```bash
cp configs/tools/foldx.example.yaml configs/tools/foldx.yaml
nano configs/tools/foldx.yaml
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --smoke-run --max-jobs 1
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --run --workers 4 --resume
```

Materialize placeholder job folders and pending `job_manifest.json` files only when explicitly requested:

```bash
python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name foldx_experimental --dry-run --materialize-plan
```

This still does not create `parsed_result.json` and does not launch FoldX/Rosetta/ESM2/OpenFold.

Rosetta experimental has the same safe control pattern:

```bash
cp configs/tools/rosetta.example.yaml configs/tools/rosetta.yaml
nano configs/tools/rosetta.yaml
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --dry-run
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --preflight
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --smoke-run --max-jobs 1
python3 scripts/executors/run_rosetta_experimental.py --dataset current_internal --run --workers 4 --resume
```

ESM2 GPU optional uses the same executor contract under `gpu_heavy`:

```bash
cp configs/tools/esm2.example.yaml configs/tools/esm2.yaml
nano configs/tools/esm2.yaml
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --dry-run
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --preflight
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --smoke-run --max-jobs 1
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --run --workers 1 --resume
```

OpenFold WT generation planning:

```bash
cp configs/tools/openfold.example.yaml configs/tools/openfold.yaml
nano configs/tools/openfold.yaml
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --dry-run
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --preflight
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --materialize-plan --batch-size 2 --force
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --smoke-run --max-batches 1 --batch-size 1 --force
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --run --batch-size 1 --workers 1 --force
```

OpenFold execution is explicit only. Dry-run, preflight, and materialize-plan never run OpenFold and never create completion markers. Real OpenFold runs create `data/run_state/current_internal/gpu_heavy.openfold_wt_generation.completed.json`, not the global `gpu_heavy.completed.json`.

## Parallel heavy dry-runs

Start CPU and GPU heavy dry-runs in separate tmux sessions:

```bash
bash scripts/run/run_parallel_heavy_tmux.sh
```

Attach:

```bash
tmux attach -t openfold3_cpu_heavy
tmux attach -t openfold3_gpu_heavy
```

Logs:

```bash
tail -f logs/run_profiles/cpu_heavy_latest.log
tail -f logs/run_profiles/gpu_heavy_latest.log
```

Individual helpers:

```bash
bash scripts/run/run_cpu_heavy_tmux.sh
bash scripts/run/run_gpu_heavy_tmux.sh
```

## Bundles

CPU bundle dry-run:

```bash
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group cpu_heavy --dry-run --server-config configs/server_paths.yaml
```

GPU bundle dry-run:

```bash
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group gpu_heavy --dry-run --server-config configs/server_paths.yaml
```

Real bundle creation:

```bash
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group cpu_heavy --force
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group gpu_heavy --force
```

Bundles are written under:

```text
bundles/
```

Each bundle has:

- an unpacked directory;
- a `.tar.gz` archive;
- a manifest JSON with file paths, sizes, sha256 checksums, included categories, and excluded categories.

Machine-specific configs are excluded from bundles by default:

- `configs/server_paths.yaml`
- `configs/*local*.yaml`
- `configs/*secret*.yaml`
- `configs/tools/foldx.yaml`
- `configs/tools/rosetta.yaml`
- `configs/tools/openfold.yaml`

Example configs, including `configs/server_paths.example.yaml` and `configs/tools/*.example.yaml`, remain included. To include the selected server config intentionally:

```bash
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group cpu_heavy --server-config configs/server_paths.yaml --include-server-config --force
```

## Sync helpers

Dry-run upload:

```bash
bash scripts/run/sync_run_bundle.sh --dry-run bundles/current_internal_cpu_heavy_TIMESTAMP.tar.gz user@host /remote/path
```

Collect outputs dry-run:

```bash
bash scripts/run/collect_server_outputs.sh --dry-run user@host:/home/jovyan/OpenFold3_project collected_outputs/current_internal
```

The collection helper copies:

- `results/`
- `reports/`
- `logs/`
- `data/run_state/`
- selected `data/processed/current_internal/`

It does not delete remote files. It refuses to collect into a non-empty local directory unless `--force` is used.

## Post-GPU CPU Heavy

After both heavy stages are complete:

```bash
python3 scripts/run/run_stage_group.py --stage-group post_gpu_cpu_heavy --dataset current_internal --dry-run
```

In non-dry mode, post-GPU CPU work requires completed CPU and GPU markers. This is where OpenFold outputs are collected and FoldX/Rosetta-on-OpenFold jobs will run.

## Analysis Reporting

After post-GPU CPU work is complete:

```bash
python3 scripts/run/run_stage_group.py --stage-group analysis_reporting --dataset current_internal --dry-run
```

This stage is for statistics, evaluation summaries, tables, plots, and final reports. It should be read-only with respect to method execution outputs.

## What is not implemented yet

- FoldX/Rosetta on OpenFold structures.
- Parsed result builders.
- Evaluation metric calculators.
- Final statistics, plots, and report builders.
- RMSD/TM-score/lDDT/GDT_TS/contact precision runners.
- DDG and pLDDT-vs-error metric runners.
- Final report generation.
