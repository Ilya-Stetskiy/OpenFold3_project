# OpenFold3 Project

`OpenFold3_project` is the single git root for the consolidated workspace.

## Layout

- `openfold-3/`: core OpenFold3 runtime, mutation screening, DDG/testbench tooling
- `openfold_notebooks/`: notebook-facing helper layer, monitoring, benchmark wrappers
- `third_party/`: vendored source-only external tooling snapshots

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
