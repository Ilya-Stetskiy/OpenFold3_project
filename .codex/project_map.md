# Project Map

## Purpose

This is a consolidated OpenFold3 project for runtime work, mutation screening, DDG/testbench workflows, notebook helpers, GPU worker integration, FoldX worker support, Docker packaging, and analysis/reporting artifacts. It now also contains a staged DDG dataset pipeline for `current_internal`: dataset registry/fetching, normalization, structure/sequence preparation, structure mismatch audit, curated override decisions, and future evaluation task definitions. The parent workspace `/mnt/d/Proga/OpenFold_codex` also contains local Codex skills, caches, external binaries, generated outputs, and a cleaned Docker snapshot.

## Stack

- Languages: Python, shell, notebooks, Docker/YAML.
- Frameworks/libraries: OpenFold3 runtime, PyTorch/CUDA paths, notebook helper packages, outbound polling worker code, FoldX integration, lightweight CSV/YAML data pipeline scripts.
- Build/package tools: `openfold-3/pyproject.toml`, Dockerfiles, docker-compose files, shell runners.
- Tests: pytest-style Python tests under worker packages, notebook helpers, and semantic OpenFold3 tests.
- Docker/deploy: `Dockerfile.gpu-worker`, `Dockerfile.foldx`, `docker-compose.gpu-worker.yml`, `docker-compose.foldx.yml`, plus core OpenFold3 Docker files under `openfold-3/docker/`.
- CI/tooling: local scripts and smoke suites; no CI workflow was deeply inspected.

## Entry points

- `README.md` — concise overview of the consolidated project layout and server/GPU verification flow.
- `check_server_gpu.sh` — consolidated server/GPU verification suite entrypoint.
- `gpu_worker/main.py` — outbound polling worker main module.
- `gpu_worker/runner.py` — worker execution path for OpenFold3 jobs.
- `foldx_worker/cli.py` — FoldX worker command-line entrypoint.
- `openfold-3/openfold3/run_openfold.py` — core OpenFold3 CLI/runtime entrypoint.
- `openfold-3/openfold3/mutation_runner.py` — mutation screening/DDG-related runtime logic.
- `openfold_notebooks/run_tests.sh` — notebook helper test runner.
- `scripts/data/fetch_dataset_sources.py` — raw-source registration/download manager for dataset inputs; supports dry-run and manifests.
- `scripts/data/normalize_current_internal.py` — normalizes `data/raw/current_internal/results.csv` into unique mutation rows, recalc jobs, legacy archives, and QC reports.
- `scripts/data/prepare_current_internal_inputs.py` — prepares current_internal PDB chain structures, sequences, input readiness, and structure/sequence manifests.
- `scripts/data/audit_current_internal_structure_mismatches.py` — diagnoses WT residue mismatches without changing normalized tables or jobs.
- `scripts/data/create_current_internal_structure_override_template.py` — creates editable curator decisions for mismatch cases.
- `scripts/data/apply_current_internal_structure_overrides.py` — applies curator decisions into derived resolved manifests/jobs only.
- `scripts/evaluation/build_evaluation_tasks.py` — builds future evaluation task definitions from resolved jobs and metric registry; does not compute metrics.
- `scripts/run/check_server_environment.py`, `scripts/run/run_stage_group.py`, `scripts/run/build_run_bundle.py` — server execution readiness checks, dry-run stage orchestration, and bundle creation.
- `scripts/run/execution_utils.py`, `scripts/run/inspect_resolved_jobs.py`, `scripts/run/run_jobs_skeleton.py`, `scripts/run/resource_scheduler.py`, `scripts/run/run_cpu_heavy_adaptive.py` — shared execution contract utilities, resolved job inspection, dry-run/materialized-plan skeletons, and adaptive CPU-heavy resource planning.
- `scripts/executors/foldx_adapter.py`, `scripts/executors/run_foldx_experimental.py` — generic FoldX adapter/runner for `--structure-source experimental|openfold`, with dry-run, preflight, materialized-plan, smoke-run, and explicit run entrypoint.
- `scripts/executors/rosetta_adapter.py`, `scripts/executors/run_rosetta_experimental.py` — Rosetta experimental adapter plus dry-run, preflight, materialized-plan, smoke-run, and explicit run entrypoint.
- `scripts/executors/esm2_adapter.py`, `scripts/executors/esm2_score_single.py`, `scripts/executors/run_esm2_gpu.py` — ESM2 GPU optional adapter/scorer/runner with dry-run, preflight, materialized-plan, smoke-run, and explicit run entrypoint.
- `scripts/executors/openfold_adapter.py`, `scripts/executors/run_openfold_wt.py` — OpenFold WT generation executor with sequence-level jobs, batch query JSON materialization, preflight reports, explicit smoke/full run modes, output normalization to `model.cif` plus converted `model.pdb`, and per-stage marker support.
- `scripts/executors/collect_openfold_outputs.py` — post-GPU collector that reads normalized OpenFold WT outputs, extracts pLDDT summaries, and writes derived OpenFold-ready FoldX/Rosetta job tables without changing source-of-truth jobs.
- `notebooks/server_run_control_current_internal.ipynb` — safe notebook control plane for server path YAML validation, dry-run commands, disabled-by-default execution controls, and upward repo-root detection from Jupyter cwd.

## Main directories

- `openfold-3/` — core OpenFold3 runtime, mutation screening, DDG/testbench tooling, semantic tests, docs, and Docker assets.
- `openfold_notebooks/` — notebook-facing helper layer, benchmarks, monitoring, query builders, smoke scripts, and tests.
- `gpu_worker/` — outbound polling GPU worker package with schemas, client, runner, artifacts, telemetry, and tests.
- `foldx_worker/` — FoldX worker CLI and tests.
- `pipeline/` — dataset, DDG, metrics, structure, and utility helpers for analysis workflows.
- `configs/` — dataset registry, download settings, DDG schemas, column aliases, and evaluation metric registry.
- `scripts/data/` — dataset fetch, normalization, structure preparation, mismatch audit, and override helper scripts.
- `scripts/evaluation/` — future evaluation task-definition builders.
- `scripts/run/` — dry-run server stage orchestration, bundle helpers, tmux helpers, and sync/collect helpers.
- `scripts/executors/` — method-specific execution adapters/runners that implement `configs/execution_contract.yaml`.
- `docs/` — DDG dataset pipeline, fetching, current_internal normalization/preparation/audit/override, and evaluation-layer docs.
- `docker/` — worker README files and example payloads.
- `notebooks/` — project notebooks such as experiment and DDG analysis notebooks.
- `reports/` — likely generated or curated analysis reports.
- `third_party/` — vendored source-only external tooling snapshots.
- `../OpenFold3_project_docker_clean/` — cleaned Docker/GPU-worker tree with duplicated worker, OpenFold3, notebook, and Docker assets.

## Important files

- `README.md` — primary orientation document.
- `WORK_LOG.md` — project work log.
- `DATASET_README.md` — dataset-oriented documentation.
- `EXPERIMENT_NOTEBOOK_PLAN.md` — notebook/experiment planning document.
- `.gitignore` — ignore policy for caches, runtime outputs, model weights, molecules, tools, and test artifacts.
- `Dockerfile.gpu-worker` — GPU worker container definition.
- `Dockerfile.foldx` — FoldX worker container definition.
- `docker-compose.gpu-worker.yml` — GPU worker compose workflow.
- `docker-compose.foldx.yml` — FoldX worker compose workflow.
- `gpu_worker/README.md` — worker behavior, job types, required environment, and local run command.
- `openfold-3/DDG_TESTBENCH.md` — DDG testbench documentation.
- `openfold-3/pyproject.toml` — package/build metadata for core OpenFold3.
- `configs/datasets.yaml` — logical dataset registry for `current_internal`, `big_curated`, and `independent_benchmark`.
- `configs/downloads.yaml` — fetch/download defaults, manifests, logs, and background-run settings.
- `configs/evaluation_metrics.yaml` — registry of scalar DDG, global structure, local mutation-site structure, confidence/error, and future complex/interface metrics.
- `configs/run_stages.yaml` — server execution profile for `preprepare`, parallel `cpu_heavy`/`gpu_heavy`, `post_gpu_cpu_heavy`, and read-only `analysis_reporting`.
- `configs/execution_contract.yaml` — shared job/stage statuses, result layout, manifest fields, and completion marker policy for future executors.
- `configs/resource_profiles.yaml` — conservative CPU/RAM/load/disk resource profile for adaptive CPU-heavy scheduling.
- `configs/tools/foldx.example.yaml` — committable FoldX config template; local `configs/tools/foldx.yaml` is gitignored.
- `configs/tools/rosetta.example.yaml` — committable Rosetta config template; local `configs/tools/rosetta.yaml` is gitignored.
- `configs/tools/openfold.example.yaml` — committable OpenFold WT config template; local `configs/tools/openfold.yaml` is gitignored.
- `configs/server_paths.example.yaml` — example server-local tool/path config; copy to ignored `configs/server_paths.yaml`.
- `configs/schemas/ddg_mutations_v1.yaml`, `configs/schemas/ddg_results_v1.yaml`, `configs/schemas/ddg_column_aliases.yaml` — DDG table schemas and raw column aliases.
- `docs/data_pipeline.md`, `docs/data_fetching.md`, `docs/current_internal_normalization.md`, `docs/current_internal_input_preparation.md`, `docs/current_internal_structure_mismatch_audit.md`, `docs/current_internal_structure_overrides.md`, `docs/evaluation_layer.md`, `docs/server_execution.md`, `docs/execution_contract.md`, `docs/cpu_heavy_adaptive_scheduler.md`, `docs/foldx_experimental_executor.md`, `docs/rosetta_experimental_executor.md`, `docs/esm2_gpu_executor.md`, `docs/openfold_wt_executor.md`, `docs/server_run_control_notebook.md` — user-facing DDG pipeline and execution contract docs.

## Tests

- Test locations:
  - `gpu_worker/tests/`
  - `foldx_worker/tests/`
  - `openfold_notebooks/tests/`
  - `openfold-3/semantic_tests/`
  - `../OpenFold3_project_docker_clean/gpu_worker/tests/`
  - `../OpenFold3_project_docker_clean/openfold-3/semantic_tests/`
- Likely commands:
  - `bash check_server_gpu.sh --quick`
  - `bash check_server_gpu.sh`
  - `bash openfold_notebooks/run_tests.sh`
  - targeted `pytest` on specific test files when editing a narrow module.
- Tests not to run without explicit user request:
  - full GPU/server verification without `--quick`;
  - real `run_openfold predict` smoke tests;
  - end-to-end `predict + screen-mutations` smokes;
  - DDG harness and benchmark suites;
  - tests requiring CUDA, model weights, large datasets, or external binaries.
- Lightweight DDG script checks that are usually safe when those scripts are touched:
  - `python3 -m py_compile scripts/data/<script>.py`
  - `python3 scripts/data/<script>.py --dataset current_internal --dry-run`
  - `python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --dry-run`

## Generated / ignored paths

Do not read these without explicit need:

- `.git/`
- `.pytest_cache/`
- `.runtime/`
- `.runtime_codex/`
- `data/`
- `data/raw/`
- `data/structures/`
- `analysis/`
- `results/`
- `logs/`
- `runtime_smoke/`
- `molecules/`
- `tools/`
- `openfold-3/runtime_smoke/`
- `openfold_notebooks/.runtime/`
- `openfold_notebooks/.pdb_cache/`
- `../.torch-cache/`
- `../pytest-temp/`
- `../.docker-codex/`
- `../foldx5_Linux/`
- `../OpenFold3_project_docker_clean/.git/`
- `../OpenFold3_project_docker_clean/.pytest_cache/`
- any `__pycache__/`, `.ruff_cache/`, model weights, checkpoints, raw PDB datasets, result directories, logs, or generated benchmark outputs.
- DDG generated outputs under `data/interim/`, `data/processed/`, `data/manifests/`, `reports/current_internal/`, and `data/sequences/` can be read only when the task specifically concerns the dataset pipeline. Prefer headers, counts, and targeted rows instead of opening full CSV/JSON files.

## Common workflows

Likely workflows:

- Run project: set `OPENFOLD_PROJECT_DIR`, `OPENFOLD_REPO_DIR`, and related worker/OpenFold env vars; run OpenFold through `python -m openfold3.run_openfold ...` from the configured OpenFold repo context.
- Run worker locally: use `PYTHONPATH=/work/OpenFold3_project:/work/OpenFold3_project/openfold-3 python -m gpu_worker.main` with required worker/server env vars.
- Run tests: prefer targeted `pytest` on touched files; use `openfold_notebooks/run_tests.sh` for notebook helpers; use `check_server_gpu.sh --quick` for broader server/GPU verification.
- Dataset registry/fetch dry-run: `python3 scripts/data/fetch_dataset_sources.py --dataset all --dry-run`.
- Normalize current_internal: `python3 scripts/data/normalize_current_internal.py --dataset current_internal --force`.
- Prepare current_internal structures/sequences: `python3 scripts/data/prepare_current_internal_inputs.py --dataset current_internal --all --workers 8`.
- Audit and resolve current_internal structure mismatches: run mismatch audit, create/edit `data/manifests/current_internal_structure_override_template.csv`, then apply overrides to derived resolved outputs.
- Build future evaluation tasks: `python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --force`.
- Check server execution readiness: `python3 scripts/run/check_server_environment.py --stage-group all --server-config configs/server_paths.yaml`.
- Preview run stages: `python3 scripts/run/run_stage_group.py --stage-group cpu_heavy --dataset current_internal --dry-run`.
- Plan adaptive CPU-heavy workers: `python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run`.
- Build server bundles: `python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group cpu_heavy --dry-run`.
- Inspect resolved execution jobs: `python3 scripts/run/inspect_resolved_jobs.py --dataset current_internal`.
- Preview future executor jobs: `python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name foldx_experimental --dry-run`.
- FoldX experimental preflight/materialization: `python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --preflight`; `python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --materialize-plan --max-jobs 2 --force`.
- ESM2 GPU optional preflight/materialization: `python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --preflight`; `python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --materialize-plan --max-jobs 2 --force`.
- OpenFold WT preflight/materialization/execution: `python3 scripts/executors/run_openfold_wt.py --dataset current_internal --preflight`; `python3 scripts/executors/run_openfold_wt.py --dataset current_internal --materialize-plan --batch-size 2 --max-sequences 2 --force`; server-only smoke/full execution uses `--smoke-run --max-batches 1` or explicit `--run`.
- Build/deploy: use the GPU/FoldX Dockerfiles and docker-compose files under this directory or the cleaned Docker snapshot when working on deploy packaging.
- Audit/review: start with `README.md`, then narrow to the relevant package (`openfold-3`, `openfold_notebooks`, `gpu_worker`, `foldx_worker`, or `pipeline`).
- Docs: keep `README.md`, worker README files, Docker README files, and DDG/testbench docs consistent when changing user-visible workflows.

## Risk zones

Potentially risky or complex areas:

- `openfold-3/openfold3/` — runtime-critical OpenFold logic, mutation screening, DDG/testbench behavior, numerical/GPU correctness.
- `gpu_worker/runner.py` and `schemas.py` — job translation, variant handling, result packaging, runtime limits, and compatibility with the global server queue.
- `gpu_worker/client.py` — external server API interaction, leases, uploads, retries, and authentication/token behavior.
- `openfold_notebooks/` — notebook helper behavior, path resolution, monitoring, benchmark wrappers, and smoke workflows.
- `pipeline/` — dataset and metric correctness for DDG/article analysis.
- `scripts/data/` — generated-data correctness, path discipline, manifest updates, and preservation of raw/normalized source-of-truth tables.
- `scripts/evaluation/` and `configs/evaluation_metrics.yaml` — task cardinality, metric applicability, expected path conventions, and avoiding premature metric computation.
- `scripts/run/` and `configs/run_stages.yaml` — stage dependency correctness, dry-run safety, server/local path assumptions, bundle contents, and avoiding accidental heavy execution.
- `data/raw/current_internal/results.csv` — raw local pilot dataset; do not modify in normal pipeline work.
- `data/processed/current_internal/current_internal_mutations.csv` — normalized mutation source of truth; downstream override/evaluation layers must not rewrite it.
- `data/processed/current_internal/current_internal_recalc_jobs_resolved.csv` — derived execution-readiness table after structure overrides; future execution layers should prefer this over the original jobs-with-inputs once curation begins.
- Docker and compose files — CUDA/runtime dependencies, mount paths, environment variables, and binary availability.
- Generated analysis/data areas — large files and outputs can waste tokens and may not be source of truth.
- Duplicate-looking trees (`OpenFold3_project/` vs `OpenFold3_project_docker_clean/`) — verify which tree the user intends before editing.

## Recommended next checks

1. Before any specific task, confirm whether edits should target `OpenFold3_project/` or `OpenFold3_project_docker_clean/`.
2. For code work, read the narrow module plus adjacent tests first; avoid broad rescans.
3. For OpenFold/runtime changes, inspect `openfold-3/pyproject.toml`, the relevant `openfold3/` module, and semantic tests before patching.
4. For worker changes, inspect `gpu_worker/README.md`, `schemas.py`, `runner.py`, and the matching tests.
5. For current_internal data work, read `configs/datasets.yaml`, the relevant `docs/current_internal_*.md` or `docs/evaluation_layer.md`, and only the needed CSV headers/counts before editing scripts.
6. For documentation changes, check whether README, DDG pipeline docs, and worker/Docker docs need synchronized updates.

## Notes for Codex

- Use this file before broad project analysis.
- Do not rescan the whole repository unless this map is stale or the user explicitly asks.
- Prefer narrow working sets.
- Prefer `rg`/`grep` over opening many files.
- Parent `/mnt/d/Proga/OpenFold_codex` is a workspace wrapper; this directory is the primary project root described by the project README.
- Treat voice-transcribed requests carefully; filenames and technical terms may need verification against this map and the filesystem.
