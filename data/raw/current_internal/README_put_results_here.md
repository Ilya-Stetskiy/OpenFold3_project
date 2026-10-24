# Manual source setup: local_results_csv

Dataset: `current_internal`
Dataset role: `pilot_internal_dataset`
Source type: `local_results`
Download mode: `local_file`
Required: `True`

Put the file(s) here:

```text
data/raw/current_internal/results.csv
```

Expected filename patterns:

- no specific pattern

Notes:

Local pilot file with experimental DDG and already computed method results. The fetch manager only registers this file; it does not download current_internal from the internet.

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset current_internal
```
