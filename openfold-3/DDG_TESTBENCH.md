# ddG Testbench

This repository now contains an initial `openfold3.testbench` scaffold for comparing ddG methods over shared structural inputs.

## Current scope

- shared `BenchmarkCase` input model
- local report generation through `DdgBenchmarkHarness`
- working local FoldX integration through `BuildModel + AnalyseComplex`
- first local Rosetta integration through `score_jd2`
- SQLite registry for runs, cases, method results, and per-method stage timings
- summary/evaluation layer over stored reports
- CLI entrypoints for single-case and multi-case execution

## Main entrypoints

- `scripts/dev/run_ddg_benchmark_harness.py`
  Runs one case and prints a single JSON report.

- `scripts/dev/run_ddg_testbench.py`
  Runs one or many cases, writes per-case reports, a SQLite registry, and an evaluation summary.

- `scripts/dev/summarize_ddg_testbench.py`
  Builds an aggregate summary from previously written report JSON files.

- `scripts/dev/run_multiscale_ddg_testbench.py`
  Runs the mutation-centric multiscale ddG benchmark with the five primary
  method families only, without the extra server-only wrappers.

- `scripts/dev/run_panel_ddg_stand.py`
  Runs the WT-plus-19-mutants panel stand over one or many mutable positions
  and persists the staged panel state in SQLite.

- `scripts/dev/summarize_panel_ddg_stand.py`
  Builds consensus/ranking exports from a completed `state.sqlite` panel run.

- `scripts/dev/run_local_mutation_benchmark.py`
  Runs the local FoldX mutation benchmark over known WT/mutant reference pairs
  plus cyclic round-trip cases, and writes RMSD-in-Angstrom summaries.

## Single-case example

```bash
PYTHONPATH=/path/to/openfold-3 python3 scripts/dev/run_ddg_testbench.py \
  --output-root runtime_smoke/ddg_testbench_single \
  --dataset-kind benchmark \
  --case-id sample_case \
  --structure /absolute/path/to/sample_case.pdb \
  --mutation A:L42A \
  --experimental-ddg -1.2
```

## Batch example

Use `examples/example_testbench_cases.json` as a template:

```bash
PYTHONPATH=/path/to/openfold-3 python3 scripts/dev/run_ddg_testbench.py \
  --output-root runtime_smoke/ddg_testbench_batch \
  --dataset-kind benchmark \
  --cases-json examples/example_testbench_cases.json
```

## Output layout

- `run_manifest.json`
- `evaluation_summary.json`
- `registry.sqlite`
- `reports/<case_id>.json`

## Panel Stand

The repository also contains a position-wise panel runner that expands one WT
query into mutation panels and tracks MSA, prediction, and ddG-analysis status
through a `state.sqlite` database.

Single-target example:

```bash
PYTHONPATH=/path/to/openfold-3 python3 scripts/dev/run_panel_ddg_stand.py \
  --target-id spike_demo \
  --wt-query-json /absolute/path/to/wt_query.json \
  --output-root runtime_smoke/panel_ddg_stand \
  --mutable-chain-id A \
  --positions 614,615
```

Summary export:

```bash
PYTHONPATH=/path/to/openfold-3 python3 scripts/dev/summarize_panel_ddg_stand.py \
  --run-root runtime_smoke/panel_ddg_stand
```

## Current limitations

- FoldX score is currently derived from `AnalyseComplex(mutant) - AnalyseComplex(wt)` and is treated as a binding ddG proxy.
- Rosetta currently uses `score_jd2` as a structural energy proxy; production ddG protocols such as `flex_ddG` or `cartesian_ddg` are not wired yet.
- CIF inputs are accepted by the harness and converted to PDB internally for FoldX and Rosetta.
- The current testbench runner is report-oriented and registry-backed, but not yet the full staged `CPU-prep -> GPU -> CPU-ddG` scheduler.

## Local tools

- FoldX is expected at `tools/bin/foldx`, `PATH`, or `FOLDX_BINARY`.
- Rosetta `score_jd2` is expected through `ROSETTA_SCORE_JD2_BINARY`, `PATH`, or a local Rosetta bundle under `tools/`.
- Rosetta `database` is expected through `ROSETTA_DATABASE` or a local Rosetta bundle under `tools/`.

If only the Rosetta archive is available, use:

```bash
python3 scripts/dev/install_rosetta_bundle_subset.py \
  --archive /mnt/d/Download/rosetta_binary_ubuntu_3.15_bundle.tar.bz2 \
  --install-root tools/rosetta3.15_min
```
