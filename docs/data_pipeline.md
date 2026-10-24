# DDG dataset pipeline

This project keeps DDG datasets separated by role and lifecycle stage. The goal is to make new datasets configurable through `configs/datasets.yaml` without changing project structure or analysis code.

No data is downloaded by this setup. The directories and config only define where source files, cleaned tables, predictions, and reports should live.

## Directory stages

`raw` contains original source files exactly as imported or received. Do not rewrite raw data in place. If a source archive, CSV, PDB bundle, or database export is needed later, store it under `data/raw/{dataset_name}`.

`interim` contains temporary normalized files produced while parsing raw sources. These files may still be source-specific and are useful for debugging parsers, joins, and structure mapping.

`processed` contains canonical dataset tables ready for analysis. Processed mutation tables should follow `configs/schemas/ddg_mutations_v1.yaml`. Method result tables should follow `configs/schemas/ddg_results_v1.yaml`.

`results` contains method outputs and analysis-ready prediction tables for one dataset. Keep method-specific raw outputs here rather than mixing them with input data.

`reports` contains figures, summaries, and dataset-specific report artifacts.

`data/manifests` contains optional dataset manifests with provenance, source versions, row counts, filtering decisions, and locks for benchmark releases.

## Dataset roles

`current_internal` is the existing local pilot dataset. It already contains experimental DDG values and computed FoldX, Rosetta, ESM2, and OpenFold results. It is useful for internal sanity checks, pipeline development, and pilot analysis. It must not be presented as an independent benchmark.

`big_curated` is the planned large curated dataset from ThermoMutDB and ProThermDB. Its role is expansion and robustness analysis. It can be used to debug the pipeline and test whether conclusions are stable across a larger data source.

`independent_benchmark` is the planned final external benchmark from S669 and/or S461. It is isolated so the final comparison is not biased by earlier tuning decisions.

## Why the independent benchmark is locked

The independent benchmark must not be used for tuning, debugging claims, threshold selection, feature selection, or method selection. If the pipeline is repeatedly adjusted after looking at this dataset, the final benchmark becomes part of development and no longer measures external generalization.

Allowed use is limited to locked final evaluation and final benchmark reporting. Any issue discovered during final evaluation should be documented; fixes should be validated on `current_internal` or `big_curated` first, then the independent benchmark should be rerun under a clearly versioned protocol.

## Adding a dataset

1. Add a new entry under `datasets` in `configs/datasets.yaml`.
2. Use a lowercase identifier with underscores, for example `new_dataset_name`.
3. Define `raw_dir`, `interim_dir`, `processed_dir`, `results_dir`, and `reports_dir`. Paths must be relative to the project root.
4. Define one or more `sources` entries with a nested `download` block.
5. Run:

```bash
python3 scripts/data/validate_dataset_config.py
python3 scripts/data/init_dataset_dirs.py --dataset new_dataset_name
```

No script should need changes for a new dataset if `configs/datasets.yaml` contains the complete entry.

## Required config fields

Each dataset entry must define:

- `display_name`
- `status`
- `role`
- `description`
- `raw_dir`
- `interim_dir`
- `processed_dir`
- `results_dir`
- `reports_dir`
- `allowed_for_tuning`
- `allowed_for_final_benchmark`
- `sources`

Each source must include `name`, `source_type`, and a nested `download` block with:

- `mode`
- `urls`
- `expected_path`
- `expected_filename_patterns`
- `required`
- `allow_parallel`
- `checksum`
- `notes`

## Required mutation fields

Processed mutation tables must include at least:

- `dataset_id`
- `mutation_id`
- `protein_id`
- `mutation`
- `wild_type`
- `mutant`
- `position`
- `experimental_ddg`

Recommended provenance and experimental-context fields are listed in `configs/schemas/ddg_mutations_v1.yaml`.

## Required result fields

Prediction result tables must include at least:

- `dataset_id`
- `mutation_id`
- `method`
- `run_id`
- `predicted_ddg`
- `status`

Rows for failed, skipped, or missing predictions should still be represented with a non-`ok` `status` so coverage and failure rates remain measurable.

## Utility scripts

List configured datasets:

```bash
python3 scripts/data/list_datasets.py
```

Validate config structure and schema references:

```bash
python3 scripts/data/validate_dataset_config.py
```

Create directories declared in the config:

```bash
python3 scripts/data/init_dataset_dirs.py
```

Plan future raw-source fetching without downloading data:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset all --dry-run
```
