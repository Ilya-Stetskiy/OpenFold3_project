# OpenFold3 Project

`OpenFold3_project` is the single git root for the consolidated workspace.

## Layout

- `openfold-3/`: core OpenFold3 runtime, mutation screening, DDG/testbench tooling
- `openfold_notebooks/`: notebook-facing helper layer, monitoring, benchmark wrappers
- `configs/`: project configuration, including DDG dataset registry and schemas
- `scripts/data/`: lightweight utilities for DDG dataset registry validation and directory setup
- `docs/`: project workflow documentation
- `third_party/`: vendored source-only external tooling snapshots

## DDG Data Pipeline

Dataset roles, directory stages, required fields, and benchmark-use rules for DDG method comparison are documented in `docs/data_pipeline.md`. Future raw-source fetching is documented in `docs/data_fetching.md`; current internal normalization is documented in `docs/current_internal_normalization.md`. Server execution staging is documented in `docs/server_execution.md`; the shared executor result contract is documented in `docs/execution_contract.md`. FoldX, Rosetta, ESM2, and OpenFold WT executor layers are documented in `docs/foldx_experimental_executor.md`, `docs/rosetta_experimental_executor.md`, `docs/esm2_gpu_executor.md`, and `docs/openfold_wt_executor.md`; notebook-based server path validation is documented in `docs/server_run_control_notebook.md`.

The dataset registry lives in `configs/datasets.yaml`. Add new logical datasets there, then validate and initialize directories with:

```bash
python3 scripts/data/validate_dataset_config.py
python3 scripts/data/init_dataset_dirs.py
```

## DDG Server Execution

The `current_internal` run profile is split into four logical stages:

- `preprepare`
- `cpu_heavy`
- `gpu_heavy`
- `postprocess`

After `preprepare`, CPU-heavy FoldX/Rosetta work and GPU-heavy OpenFold/ESM2 work can run in parallel. `postprocess` waits for both heavy stages. The stage-group orchestrator is dry-run/planning only; method-specific executors require explicit `--run` or `--smoke-run`:

```bash
python3 scripts/run/check_server_environment.py --stage-group all
python3 scripts/run/run_stage_group.py --stage-group preprepare --dataset current_internal --dry-run
python3 scripts/run/inspect_resolved_jobs.py --dataset current_internal
python3 scripts/run/run_jobs_skeleton.py --dataset current_internal --stage-name foldx_experimental --dry-run
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run
python3 scripts/executors/run_foldx_experimental.py --dataset current_internal --preflight
python3 scripts/executors/run_esm2_gpu.py --dataset current_internal --preflight
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --preflight
bash scripts/run/run_parallel_heavy_tmux.sh
```

## Path Model

- `OPENFOLD_PROJECT_DIR` defaults to the detected `OpenFold3_project` root
- `OPENFOLD_REPO_DIR` defaults to `OpenFold3_project/openfold-3`
- notebook helpers resolve relative paths from the project root and can still be overridden via env vars

## Notes

- Nested git repositories and detached worktrees were removed from the consolidated tree.
- Local caches, runtime outputs, checkpoints, and test artifacts are intentionally excluded from git.
- Large third-party model weights are not committed; only source snapshots are kept under `third_party/`.

## Server GPU Check

Run the consolidated server/GPU verification suite from the repo root:

```bash
bash ./check_server_gpu.sh
```

What it covers:

- CUDA visibility through `nvidia-smi` and `torch.cuda`
- merged `openfold-3` mutation-runner tests
- semantic CUDA smoke tests
- kernel regression tests
- notebook/helper test pack
- real `run_openfold predict` smoke
- real end-to-end `predict + screen-mutations` smoke
- comparison wrapper smoke for batch predict vs screening

Useful options:

```bash
bash ./check_server_gpu.sh --quick
bash ./check_server_gpu.sh --with-ddg-harness
```
