# OpenFold3 GPU Worker

This package implements an outbound-only polling worker for a global OpenFold3 web
service. The global server owns the queue; the worker leases one job at a time,
runs the existing OpenFold3 CLI, packages the full output, uploads it to the
server-provided upload target, and keeps a local TTL cache for debugging.

## Job Types

- `predict_batch`: one or more independent high-level query specs. The worker
  writes `query.json` and runs `python -m openfold3.run_openfold predict`.
- `variant_batch`: one base query plus variants. Single point-mutation variants
  are routed through `screen-mutations`; arbitrary variants are converted into a
  multi-query `predict` payload.

## Required Environment

- `WORKER_ID`
- `WORKER_TOKEN`
- `SERVER_URL`
- `OPENFOLD_PROJECT_DIR`
- `OPENFOLD_REPO_DIR`
- `OPENFOLD_PREFIX`
- `WORKER_RESULTS_DIR`
- `WORKER_CACHE_TTL_DAYS`
- `MAX_JOB_RUNTIME_SECONDS`
- `MIN_FREE_DISK_GB`

Run locally:

```bash
PYTHONPATH=/work/OpenFold3_project:/work/OpenFold3_project/openfold-3 \
python -m gpu_worker.main
```
