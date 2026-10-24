# Manual source setup: prothermdb

Dataset: `big_curated`
Dataset role: `expansion_robustness_analysis`
Source type: `database_export`
Download mode: `manual_download`
Required: `False`

Put the file(s) here:

```text
data/raw/big_curated/prothermdb
```

Expected filename patterns:

- `*.csv`
- `*.tsv`
- `*.xlsx`

Notes:

Manual source. Put exported ProThermDB files in the source raw directory and record provenance in the source manifest.

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset big_curated
```
