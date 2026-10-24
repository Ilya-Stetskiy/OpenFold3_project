# Manual source setup: s461

Dataset: `independent_benchmark`
Dataset role: `benchmark`
Source type: `benchmark`
Download mode: `manual_download`
Required: `False`

Put the file(s) here:

```text
data/raw/independent_benchmark/s461
```

Expected filename patterns:

- `*.tsv`
- `*.csv`
- `*.xlsx`

Notes:

Optional manual benchmark source. Keep it separated from big_curated and never use it for pipeline tuning.

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset independent_benchmark
```
