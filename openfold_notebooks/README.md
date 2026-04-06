# OpenFold Notebook Project

This folder contains a notebook-first wrapper around OpenFold for external users.

Project layout:

```text
openfold_notebooks/
  helpers/
    of_notebook_lib/
      __init__.py
      analysis.py
      config.py
      query_builders.py
      runner.py
      screening.py
      workflows.py
  01_single_complex.ipynb
  02_mutation_batch.ipynb
  OpenFold3_Length_Benchmark.ipynb
  OpenFold3_Runtime_Benchmark.ipynb
```

What each notebook is for:

- `01_single_complex.ipynb`: run one protein or complex prediction and inspect the best samples.
- `02_mutation_batch.ipynb`: run a set of point mutations against one base complex and rank the results.
- `OpenFold3_Length_Benchmark.ipynb`: run RMSD-vs-length experiments on real PDB inputs.
- `OpenFold3_Runtime_Benchmark.ipynb`: profile cold/warm OpenFold3 runtime, CPU/GPU load, and per-case timelines across protein lengths.

Advanced batch modes on `main`:

- notebook `predict` batch: one `run_openfold predict` call with multiple mutated queries
- screening batch: `run_openfold screen-mutations` with cached per-mutation orchestration from the sibling `openfold-3` checkout
- comparison wrapper: run both approaches and write a timing summary with speedup ratio

Design goals:

- users edit only a few cells with sequences and run settings
- helper code lives under `helpers/`
- OpenFold runtime paths are configured in one place
- result tables are prepared automatically
- notebook-facing imports are kept intentionally small; internal helper modules stay available for development and tests

Important limitations:

- OpenFold confidence metrics are not direct `ddG` estimates
- `ipTM` is useful for ranking interface candidates, not for binding affinity claims
- mutation ranking here is a screening step, not a final biophysical conclusion

Typical usage:

1. Open one notebook.
2. Edit the input cell with molecules and options.
3. Adjust `RuntimeConfig` if your OpenFold environment lives somewhere else.
4. Run all cells.

Notebook UX:

- both notebooks have one main user-edit cell
- both notebooks show an input preview before launch
- both notebooks write `query.json`, `run_openfold.log`, raw outputs, and a `summary/` folder
- the single-complex notebook shows a compact best-sample table and a quick interpretation
- the mutation-batch notebook shows per-sample output, per-mutation summary, and final mutation ranking
- advanced helper API also exposes `run_screened_mutation_case()`, `compare_mutation_batch_case()`, and `run_server_end_to_end_case()`
- a local low-memory runner config is available at `./configs/low_mem.yml`
- runtime profiling artifacts now live under `openfold3_runtime_benchmark/runs/<timestamp>/` with `manifest.json`, `case_results.csv`, `events.jsonl`, `samples.jsonl`, SVG plots, and per-case timeline summaries

## Advanced Screening API

For the mutation-screening stage runner, `openfold_notebooks/main` now includes thin wrappers around the sibling `openfold-3` checkout.

Default sibling checkout location:

```text
../openfold-3
```

Override if needed:

```bash
export OPENFOLD_REPO_DIR=/absolute/path/to/openfold-3
```

Notebook-facing entry points:

- `run_screened_mutation_case(...)`
- `compare_mutation_batch_case(...)`
- `run_server_end_to_end_case(...)`

What comparison mode writes:

- wall-clock time for notebook `predict` batch
- wall-clock time for `screen-mutations`
- `time_saved_seconds`
- `speedup_ratio`
- internal `screen-mutations` totals from `results.jsonl`

## Test Pack

This project includes a self-contained test pack for the notebook helper layer.

Included:

- unit tests for query building and validation
- analysis tests on bundled fixture output in `tests/fixtures/openfold_output`
- mocked runtime tests for `runner.py`, including `run_cmd()` and almost all of `run_prediction()`
- benchmark-package tests for `openfold3_length_benchmark` and `openfold3_runtime_benchmark`
- coverage report in the default test command

One-command test run on Windows / PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
```

One-command test run on Linux:

```bash
bash ./run_tests.sh
```

What the command does:

1. installs test dependencies into local folder `.test_deps`
2. sets `PYTHONPATH` to local deps and `helpers/`
3. runs `pytest` with coverage for `of_notebook_lib`

Current scope:

- helper layer is tested
- notebook-facing API is tested
- real `OpenFold` inference is not executed by the test pack

This means the tests validate almost all wrapper logic, but they do not confirm that your server's OpenFold runtime, weights, CUDA, MSA cache, and external binaries are fully working together.

## Server Smoke Test

For a real server-side runtime check, use the Linux smoke test:

```bash
bash ./run_server_smoke.sh
```

What it does:

- runs real `run_openfold predict`
- uses a tiny single-protein ubiquitin query from `server_smoke/query_ubiquitin.json`
- disables templates
- uses `num_diffusion_samples=1` and `num_model_seeds=1`
- checks that at least one `*_confidences_aggregated.json` file and one model file are produced

If `run_openfold` is not in `PATH`, set:

```bash
export OPENFOLD_RUNNER=/absolute/path/to/run_openfold
```

Optional custom output directory:

```bash
bash ./run_server_smoke.sh /tmp/openfold_smoke_output
```

This smoke test is intentionally separate from the normal unit/integration test pack. It validates the actual OpenFold runtime on the server.

## Server End-to-End Smoke

For a broader server run that covers both single prediction and mutation screening:

```bash
bash ./run_server_end_to_end.sh
```

This command:

- probes `nvidia-smi` if available
- runs a minimal single-protein predict case
- runs a minimal `screen-mutations` case on the same protein unless `--single-only` is passed
- writes one JSON summary path at the end
