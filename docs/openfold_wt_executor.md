# OpenFold WT executor

This layer prepares and executes OpenFold WT generation for the `current_internal` dataset.

By default it is non-executing: dry-run, preflight, and materialize-plan do not run OpenFold. Real execution requires an explicit `--smoke-run` or `--run`.

## Scope

The executor works on unique chain sequences from:

```text
data/manifests/current_internal_sequence_manifest.csv
```

Logical tracking is per `sequence_id`, while execution planning supports batches. This avoids creating 655 duplicate WT structure jobs for mutation rows that share the same PDB chain sequence.

## Configure

Create a local machine-specific config:

```bash
cp configs/tools/openfold.example.yaml configs/tools/openfold.yaml
nano configs/tools/openfold.yaml
```

`configs/tools/openfold.yaml` is gitignored.

Important fields:

- `openfold.python_bin`
- `openfold.openfold_repo_dir`
- `openfold.runner_yaml`
- `openfold.inference_ckpt_path`
- `openfold.inference_ckpt_name`
- `openfold.num_diffusion_samples`
- `openfold.num_model_seeds`
- `openfold.use_msa_server`
- `openfold.use_templates`
- `openfold.batch_size`

For article-scale runs, avoid accidental downloads or external network behavior by setting checkpoint and MSA/template behavior explicitly.

## Commands

Dry-run:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --dry-run
```

Preflight:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --preflight
```

Materialize batch plans without running OpenFold:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --materialize-plan --batch-size 2 --force
```

Limit to one sequence for debugging:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --materialize-plan --max-sequences 1 --force
```

Smoke-run one batch on the server:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --smoke-run --max-batches 1 --batch-size 1 --force
```

Full server run:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --run --batch-size 1 --workers 1 --force
```

Keep `--workers 1` unless the server has enough GPU memory to run multiple independent OpenFold processes. `--batch-size` controls how many sequence queries go into one OpenFold invocation.

## Outputs

Per-sequence job directories:

```text
results/current_internal/openfold/wt/{sequence_id}/
  model.cif
  model.pdb
  confidence.json
  openfold_result.json
  job_manifest.json
```

Batch planning directories:

```text
results/current_internal/openfold/wt/_batches/batch_0001/
  query.json
  command.txt
  batch_manifest.json
  openfold_output/
```

Stage manifest:

```text
results/current_internal/_manifests/openfold_wt_generation_jobs.csv
results/current_internal/_manifests/openfold_wt_generation_jobs.json
```

No completion marker is created by dry-run, preflight, or materialize-plan.

For real `--smoke-run` or `--run`, the executor creates only a per-stage marker:

```text
data/run_state/current_internal/gpu_heavy.openfold_wt_generation.completed.json
```

It does not create the global `gpu_heavy.completed.json`; that marker belongs to the stage-group orchestration after all GPU-heavy stages are complete.

## Output Normalization

After OpenFold finishes, the executor reads `summary.jsonl`, selects the highest ranked sample, and copies the selected files into the standard per-sequence layout:

- `model.cif`
- `model.pdb`
- `confidence.json`
- `openfold_result.json`

`model.pdb` is converted from the selected `model.cif` during normalization so downstream FoldX/Rosetta preparation can use a stable PDB path. If the model, confidence file, or CIF-to-PDB conversion cannot be completed, the sequence job is marked `parse_failed` instead of inventing a result.

Server smoke-runs must explicitly verify this conversion on a real OpenFold output before a full run:

```bash
python3 scripts/executors/run_openfold_wt.py --dataset current_internal --smoke-run --max-batches 1 --batch-size 1 --force
python3 scripts/executors/collect_openfold_outputs.py --dataset current_internal --dry-run
```

The collector should report the smoke sequence as ready only when `model.cif`, `model.pdb`, and `confidence.json` are all present.
