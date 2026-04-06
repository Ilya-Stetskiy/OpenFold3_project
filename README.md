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
