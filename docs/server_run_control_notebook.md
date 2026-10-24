# Server run control notebook

`notebooks/server_run_control_current_internal.ipynb` is a safe notebook control plane for checking server paths and previewing current_internal server commands.

The notebook is non-destructive by default. It does not run FoldX, Rosetta, OpenFold, ESM2, rsync, or metrics unless explicit flags are changed.

## Working with configs/server_paths.yaml from the notebook

The notebook uses:

- `SERVER_CONFIG = "configs/server_paths.yaml"`
- `SERVER_CONFIG_EXAMPLE = "configs/server_paths.example.yaml"`

`configs/server_paths.yaml` is gitignored and should contain local/server-specific paths. The example file is versioned and should stay generic. The notebook searches upward from the current Jupyter directory to find the repository root, using `configs/datasets.yaml` as the main marker, then resolves relative paths from that detected root.

Create the local config manually:

```bash
cp configs/server_paths.example.yaml configs/server_paths.yaml
nano configs/server_paths.yaml
```

## Safety flags

Defaults:

```python
AUTO_CREATE_SERVER_CONFIG = False
SAVE_SERVER_CONFIG_FROM_NOTEBOOK = False
CONFIRM_WRITE_SERVER_CONFIG = False
```

Meaning:

- `AUTO_CREATE_SERVER_CONFIG`: when `True`, the notebook may copy the example YAML to `configs/server_paths.yaml`.
- `SAVE_SERVER_CONFIG_FROM_NOTEBOOK`: when `True`, the notebook may write non-`None` overrides into `configs/server_paths.yaml`.
- `CONFIRM_WRITE_SERVER_CONFIG`: second confirmation required before any YAML write.

Writing is refused unless both `SAVE_SERVER_CONFIG_FROM_NOTEBOOK=True` and `CONFIRM_WRITE_SERVER_CONFIG=True`. If `configs/server_paths.yaml` is missing, writing is also refused unless `AUTO_CREATE_SERVER_CONFIG=True`.

## Notebook overrides

`None` means use the YAML value. Non-`None` means an explicit notebook override. Overrides are applied only when values are not `None`:

- `SERVER_PROJECT_ROOT`
- `SERVER_DATA_ROOT`
- `SERVER_RESULTS_ROOT`
- `SERVER_LOGS_ROOT`
- `SERVER_REPORTS_ROOT`
- `FOLDX_BINARY`
- `ROSETTA_BIN_DIR`
- `OPENFOLD_ENV`
- `PYTHON_BIN`
- `CUDA_VISIBLE_DEVICES`
- `MAX_PARALLEL_OPENFOLD_JOBS`

If a value is `None`, the YAML value is used. `PYTHON_BIN` and `MAX_PARALLEL_OPENFOLD_JOBS` default to `None`; command cells fall back to `python3` only for runtime command construction if the YAML also lacks `tools.python_bin`. That fallback is not written back to YAML unless the user explicitly sets a non-`None` override and enables both write safety flags.

## Path validation

The notebook validates and displays:

- `server.project_root`
- `server.data_root`
- `server.results_root`
- `server.logs_root`
- `server.reports_root`
- `tools.foldx_binary`
- `tools.rosetta_bin_dir`
- `tools.openfold_env`
- `tools.python_bin`
- `gpu.cuda_visible_devices`
- `gpu.max_parallel_openfold_jobs`
- `runtime.default_workers_cpu`
- `runtime.default_workers_gpu`
- `runtime.use_tmux`
- `runtime.tmux_session_prefix`

The output table includes:

- `field`
- `value`
- `status`
- `kind`
- `writable`
- `executable`
- `notes`

If pandas is installed, the notebook shows a DataFrame. Otherwise it prints a markdown table.

If repository root detection fails, the notebook stops with a clear error and refuses to create or write `configs/server_paths.yaml`.

## CLI commands from notebook

Notebook command cells pass `--server-config configs/server_paths.yaml` to scripts that support it:

```bash
python3 scripts/run/check_server_environment.py --stage-group all --server-config configs/server_paths.yaml
python3 scripts/run/run_stage_group.py --stage-group cpu_heavy --dataset current_internal --dry-run --server-config configs/server_paths.yaml
python3 scripts/run/build_run_bundle.py --dataset current_internal --stage-group cpu_heavy --dry-run --server-config configs/server_paths.yaml
```

FoldX controls are also present, but the full batch flag is off by default:

```python
RUN_FOLDX_FULL = False
RUN_OPENFOLD = False
```

Use FoldX preflight first, then a one-job smoke-run on the server before any full batch run.

## Bundles

Machine-specific configs are not included in bundles by default. This includes:

- `configs/server_paths.yaml`
- `configs/*local*.yaml`
- `configs/*secret*.yaml`
- `configs/tools/foldx.yaml`
- `configs/tools/rosetta.yaml`
- `configs/tools/openfold.yaml`

To include the selected server config intentionally, pass `--include-server-config` to `scripts/run/build_run_bundle.py`.
