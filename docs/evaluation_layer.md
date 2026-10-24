# Evaluation layer

The evaluation layer defines future metric tasks for current_internal and later datasets.

It is separate from execution jobs. Execution jobs produce method outputs such as FoldX, Rosetta, ESM2, or OpenFold files. Evaluation tasks consume those outputs later and compute scalar, structural, confidence, and relationship metrics.

This stage does not compute metrics and does not run OpenFold, FoldX, Rosetta, or ESM2.

## Inputs

```text
configs/datasets.yaml
configs/evaluation_metrics.yaml
data/processed/current_internal/current_internal_mutations.csv
data/processed/current_internal/current_internal_recalc_jobs_resolved.csv
data/manifests/current_internal_structure_manifest_resolved.csv
data/manifests/current_internal_sequence_manifest.csv
```

## Outputs

```text
data/processed/current_internal/current_internal_evaluation_tasks.csv
reports/current_internal/current_internal_evaluation_tasks_qc.md
reports/current_internal/current_internal_evaluation_tasks_qc.json
```

## Metric groups

`scalar_ddg` contains DDG regression against experiment and paired score comparison tasks.

`global_structure` contains global reference-versus-predicted structure metrics such as CA RMSD, backbone RMSD, all-atom RMSD, TM-score, and GDT_TS. For current_internal these are built once per unique PDB chain/sequence, not once per mutation.

`local_mutation_structure` contains mutation-site structure metrics such as local CA RMSD, local backbone RMSD, local lDDT, and local contact precision. These are built per mutation.

`confidence_error` contains pLDDT summaries and future pLDDT-versus-error relationship tasks.

`future_complex_interface` is reserved for complex/interface metrics such as DockQ, interface RMSD, and interface contact precision. These are not active for current_internal because current_internal is currently treated as a single-chain dataset.

## Adding a metric

Add a metric entry to `configs/evaluation_metrics.yaml` with:

- `metric_name`
- `metric_group`
- `metric_type`
- required and optional inputs
- outputs
- applicability flags
- blocking conditions
- notes

Then update the task builder only if the metric needs a new task generation pattern. Metrics that fit an existing group pattern can reuse the same input conventions.

## Path conventions

Future OpenFold WT structure:

```text
results/current_internal/openfold/wt/{sequence_id}/model.cif
```

Future OpenFold confidence:

```text
results/current_internal/openfold/wt/{sequence_id}/confidence.json
```

Future parsed method result:

```text
results/current_internal/{method}/{structure_source}/{mutation_record_id}/parsed_result.json
```

Future evaluation output:

```text
results/evaluation/current_internal/{metric_group}/{metric_name}/...
```

## Build tasks

```bash
python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --dry-run
python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --force
```

Filter by group or metric:

```bash
python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --metric-group global_structure --force
python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --metric-name local_lddt --force
```

Future complex/interface placeholders are excluded by default. They can be included explicitly for registry checks:

```bash
python3 scripts/evaluation/build_evaluation_tasks.py --dataset current_internal --include-future-blocked --dry-run
```

## Reading evaluation tasks

One row in `current_internal_evaluation_tasks.csv` is one future evaluation task.

`input_ready=false` means that the task is intentionally registered but cannot run yet. Common reasons:

- `pending_method_result`
- `pending_paired_method_results`
- `pending_openfold_structure`
- `pending_confidence_file`
- `pending_upstream_metrics`

This is expected at this stage because method outputs and OpenFold WT structures are not yet generated.

For pLDDT tasks, `pending_openfold_structure` means the expected OpenFold `model.cif` is missing. `pending_confidence_file` means the model file exists but the expected `confidence.json` is still missing.

## Why interface metrics are excluded

DockQ, interface RMSD, and interface contact precision require protein-complex/interface definitions. current_internal is currently prepared as single-chain PDB targets, so creating active interface tasks would be misleading. The metrics remain in the registry as a future extension.
