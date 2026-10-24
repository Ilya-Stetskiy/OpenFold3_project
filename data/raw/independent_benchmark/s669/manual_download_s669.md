# Manual source setup: s669

Dataset: `independent_benchmark`
Dataset role: `benchmark`
Source type: `benchmark`
Download mode: `remote_or_manual`
Required: `True`

Put the file(s) here:

```text
data/raw/independent_benchmark/s669
```

Expected filename patterns:

- `*.tsv`
- `*.csv`
- `*.fasta`
- `*.zip`
- `*.tar.gz`

Notes:

Independent benchmark source. Use direct URLs when available, or place files manually. Do not use this source for tuning or debugging claims.

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset independent_benchmark
```
