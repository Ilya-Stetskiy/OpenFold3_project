# DDG data fetching

## Purpose

This layer obtains or registers raw DDG source files only. It reads `configs/datasets.yaml` and `configs/downloads.yaml`, then updates source manifests and logs.

It does not normalize ThermoMutDB, ProThermDB, S669, S461, or the internal dataset. It also does not parse mutations, download PDB/mmCIF structures, run FoldX/Rosetta/OpenFold/ESM2, calculate metrics, or merge datasets.

## Dataset Separation

`current_internal` is a local pilot/internal dataset. It is expected at `data/raw/current_internal/results.csv` and is registered as `local_file`; it is not downloaded from the internet and is not an independent benchmark.

`big_curated` is the expandable robustness dataset. ThermoMutDB and ProThermDB sources live under `data/raw/big_curated/` and may be used for pipeline debugging and robustness analysis.

`independent_benchmark` is the final external benchmark. S669 and S461 sources live only under `data/raw/independent_benchmark/`. Do not use this dataset for tuning, debugging claims, threshold selection, feature selection, or method selection.

## Source Types

`local_file` checks a configured local path, computes size and SHA256 if present, and registers it in the manifest. If the file is missing, the script writes an instruction file and records `missing`.

`direct_download` downloads direct URLs into the configured raw source directory, first as `.part`, then atomically renames to the final filename. Existing files are not overwritten unless `--force` is used.

`manual_download` never tries to bypass forms, logins, or interactive sites. It writes `manual_download_{source}.md`, then registers matching files already placed in the raw source directory.

`remote_or_manual` uses direct download when URLs are configured. If URLs are absent or download fails, it falls back to manual instructions and registers manually placed matching files when present.

## Manifest

The source manifest files are:

```text
data/manifests/source_manifest.csv
data/manifests/source_manifest.json
```

Each run updates existing records by key:

```text
dataset_name + source_name + file_label + local_path + source_url
```

Before writing, existing manifests are backed up to:

```text
data/manifests/source_manifest.csv.bak
data/manifests/source_manifest.json.bak
```

Important fields include `dataset_name`, `dataset_role`, `source_name`, `source_url`, `local_path`, `download_status`, `download_mode`, `file_size_bytes`, `sha256`, `requires_manual_download`, `manual_instruction_path`, `error_message`, `file_label`, `is_required`, `allow_parallel`, `attempts`, and `elapsed_seconds`.

`sha256` makes raw provenance reproducible. To check whether a file was registered, inspect the CSV or query the JSON for its `dataset_name`, `source_name`, and `download_status`.

## Manual Downloads

Current internal results must be placed here:

```text
data/raw/current_internal/results.csv
```

ProThermDB exports must be placed here:

```text
data/raw/big_curated/prothermdb/
```

S461 files must be placed here:

```text
data/raw/independent_benchmark/s461/
```

After placing manual files, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset all
```

## Parallel Downloads

Use `--workers` to process independent file jobs in parallel:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset all --workers 8
```

The implementation parallelizes independent source/file jobs. It does not split a single large file into Range-request chunks.

## Safe Long-Running Runs

Recommended tmux run:

```bash
bash scripts/data/run_fetch_in_tmux.sh
tmux attach -t ddg_fetch
tmux ls
tail -f logs/data_fetching/fetch_latest.log
```

The tmux helper uses the session name from `configs/downloads.yaml`, defaulting to `ddg_fetch`. If the session already exists, it prints attach and log commands instead of starting another run.

## nohup Alternative

Run in the background without tmux:

```bash
bash scripts/data/run_fetch_nohup.sh
tail -f logs/data_fetching/fetch_nohup.log
```

## Dry Run

List datasets and sources:

```bash
python3 scripts/data/fetch_dataset_sources.py --list
```

Print the planned jobs without downloading data and without changing manifests:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset all --dry-run
```

Plan a single source:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset big_curated --source thermomutdb --dry-run
python3 scripts/data/fetch_dataset_sources.py --dataset independent_benchmark --source s669 --dry-run
```

## Re-run and Resume

The fetch layer is safe to rerun. If a final file already exists and checksum is absent or matches, it is registered without downloading again. Stale `.part` files are removed and logged before a new attempt. Use `--force` to redownload or reregister instead of reusing an existing ready file.

Network downloads use `--timeout` and `--retries`. Exhausted retries are logged and recorded in the manifest while other jobs continue.

## Troubleshooting

Missing local file: place the expected file at the path shown in the generated instruction file, then rerun the script.

No direct URL: the source will be marked as requiring manual download. Put the file in the configured raw directory and rerun.

Checksum mismatch: inspect `error_message` in the manifest, replace the file or fix the expected checksum, then rerun with `--force` if a redownload is needed.

Download failed: check the per-run log under `logs/data_fetching/`, then rerun. Other sources are still processed.

tmux session already exists: attach with `tmux attach -t ddg_fetch` or inspect `tail -f logs/data_fetching/fetch_latest.log`.
