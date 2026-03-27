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
      workflows.py
  01_single_complex.ipynb
  02_mutation_batch.ipynb
```

What each notebook is for:

- `01_single_complex.ipynb`: run one protein or complex prediction and inspect the best samples.
- `02_mutation_batch.ipynb`: run a set of point mutations against one base complex and rank the results.

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

## Test Pack

This project includes a self-contained test pack for the notebook helper layer.

Included:

- unit tests for query building and validation
- analysis tests on bundled fixture output in `tests/fixtures/openfold_output`
- mocked runtime tests for `runner.py`, including `run_cmd()` and almost all of `run_prediction()`
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
