# CPU-heavy adaptive scheduler

The CPU-heavy adaptive scheduler computes conservative worker counts for CPU-bound executors such as FoldX and Rosetta.

It does not replace method executors. It wraps them and passes a resource-aware `--workers` value.

## Inputs

- `configs/resource_profiles.yaml`
- `data/processed/current_internal/current_internal_recalc_jobs_resolved.csv`
- current CPU count, load average, RAM, and free disk space

## Default Behavior

By default the scheduler writes a plan only:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run
```

Reports are written to:

```text
reports/current_internal/cpu_heavy_adaptive_plan.md
reports/current_internal/cpu_heavy_adaptive_plan.json
```

No FoldX or Rosetta process is launched in dry-run mode.

## Execution Modes

Preflight method executors with adaptive worker counts:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --preflight
```

Materialize method plans:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --materialize-plan --max-jobs 2 --force
```

Run CPU-heavy executors explicitly:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --run --resume --no-progress-bar
```

For safer debugging, cap total workers:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --run --max-total-workers 4 --resume
```

Run tool stages sequentially instead of concurrently:

```bash
python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --run --sequential-tools --resume
```

## Resource Model

The scheduler uses:

- CPU core count
- current load average
- reserved CPU cores
- available RAM
- free disk space
- per-tool CPU/RAM cost estimates
- per-tool max worker caps

The default profile is conservative:

- Rosetta is allocated first because it is heavier.
- FoldX receives remaining worker capacity.
- ESM2 CPU is disabled because ESM2 currently belongs to `gpu_heavy`.

## Safety

The scheduler only launches heavy tools with explicit `--run`. The stage-group orchestrator remains dry-run only.

Per-method executors still own their normal safety behavior:

- no overwrite without `--force`
- resume support
- per-job manifests
- stage manifests
- per-stage completion markers

The scheduler does not create global `cpu_heavy.completed.json`.
