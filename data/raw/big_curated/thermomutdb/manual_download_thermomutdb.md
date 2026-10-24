# Manual source setup: thermomutdb

Dataset: `big_curated`
Dataset role: `expansion_robustness_analysis`
Source type: `database_export`
Download mode: `remote_or_manual`
Required: `True`

Put the file(s) here:

```text
data/raw/big_curated/thermomutdb
```

Expected filename patterns:

- `*.json`
- `*.csv`
- `*.tsv`
- `*.zip`
- `*.tar.gz`

Notes:

Use direct URLs when they become available; otherwise place a manual ThermoMutDB export in this raw source directory.

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset big_curated
```
