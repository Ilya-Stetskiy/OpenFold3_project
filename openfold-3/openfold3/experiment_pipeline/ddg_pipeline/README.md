# FoldX + Rosetta + ESM2 ddG Pipeline

## Supported Method

Current supported methods:

- `FoldX` — structure-based, physics-based
- `Rosetta ddg_monomer` — structure-based, stochastic, process-parallel
- `ESM2` — sequence-based, zero-shot masked language model scoring

This stage does not claim support for any other sequence-based, hybrid, or structure-based predictors.

## Scientific Protocol

The runtime contracts are:

### FoldX

1. validate canonical mutation input
2. resolve mutation site by real `(chain_id, residue_id)`
3. run `RepairPDB`
4. run `BuildModel`
5. parse generated mutant and WT models explicitly
6. compute per-run ddG
7. aggregate final ddG as the arithmetic mean of per-run ddG

For the local FoldX 5.1 runtime used in this project, `RepairPDB` is invoked without `--output-dir`. In this environment that is the only audited mode that consistently materializes the repaired PDB on disk.

### Rosetta ddg_monomer

1. validate canonical mutation input
2. prepare `input.pdb` and Rosetta mutfile
3. renumber the prepared Rosetta input consecutively before writing the mutfile
4. run `relax.static.linuxgccrelease`
5. launch multiple independent `ddg_monomer.static.linuxgccrelease` processes
6. parse `ddg_predictions.out` and the per-run wild-type / mutant energies
7. aggregate as `mean(top-k mutant energies) - mean(top-k wild-type energies)`

Rosetta execution is process-level parallelism only. The pipeline uses multiple independent processes rather than pretending that `ddg_monomer` scales by internal threading.

### ESM2

1. validate canonical mutation input
2. build WT and mutant sequences from the canonical record
3. run local `fair-esm` inference with pretrained `esm2_t33_650M_UR50D`
4. compute pseudo-log-likelihood for WT and mutant sequences by residue masking
5. report `ddg = logP(mutant) - logP(wildtype)`

ESM2 is executed locally through a dedicated Python interpreter with `torch` and `fair-esm` installed. It is a sequence-only backend and therefore runs exactly once per mutation record, not once per structure source.

The pipeline saves:

- `outputs/results.csv`
- `outputs/foldx_execution_trace.md`
- `outputs/foldx_model_pairing.json`
- `outputs/rosetta_environment_check.json`
- `outputs/rosetta_protocol.md`
- `outputs/esm_environment_check.json`
- `outputs/esm_model_info.json`
- `outputs/metrics.json` when enough valid predictions exist

## Required Environment

- local FoldX binary resolvable from `FOLDX_BINARY` or PATH
- Rosetta `ddg_monomer.static.linuxgccrelease`
- Rosetta `relax.static.linuxgccrelease`
- Rosetta `database/`
- Anaconda or another local Python with `torch` and `fair-esm`
- local ESM2 checkpoint cache reachable through `TORCH_HOME`
- optional `ESM2_DEVICE=auto|cuda|cuda:0|cpu`; use `cuda` to fail explicitly when GPU is unavailable
- writable working directories for FoldX temporary files
- writable working directories for Rosetta run directories
- writable working directories for ESM2 runtime files

Missing FoldX is treated as an explicit failure.
If `RepairPDB` reports success but does not materialize the repaired PDB on disk, the pipeline also fails explicitly.
Missing Rosetta binaries or database are also treated as explicit failures.
Missing ESM2 runtime, missing checkpoint, failed forward pass, or invalid numeric output are also treated as explicit failures.

## Failure Semantics

The pipeline is fail-fast:

- missing executable -> fail
- missing structure -> fail
- missing residue -> fail
- missing repaired structure after `RepairPDB` -> fail
- model pairing failure -> fail
- parsing failure -> fail
- relax failure -> fail
- any Rosetta ddg worker failure -> fail
- invalid Rosetta output -> fail
- missing ESM2 runtime or checkpoint -> fail
- failed ESM2 forward pass -> fail
- NaN ESM2 score -> fail

No benign fallback to fake support is allowed.

## Artifacts

`raw_output.json` contains the audit fields required for protocol inspection:

- original and repaired structure paths
- actual BuildModel input path
- mutation identifiers
- generated mutant and WT models
- explicit model pairing
- per-run ddG
- aggregated ddG and standard deviation
- requested and valid run counts
- backend status and backend logs path

For Rosetta, `raw_output.json` additionally contains:

- relax command and paths
- per-run stdout/stderr paths
- per-run ddG values
- per-run wild-type and mutant energies
- `ddg_raw`, `ddg`, `unit_raw`, `unit`
- normalization metadata
- max parallel jobs used

For ESM2, `raw_output.json` additionally contains:

- command and runtime environment check path
- WT and mutant sequences
- WT and mutant pseudo-log-likelihood values
- `ddg_raw`, `ddg`, `unit_raw`, `unit`
- model metadata and backend logs path

## Server Run Workflow

Long runs should use the server-run entrypoint instead of the interactive demo path:

```bash
python -m openfold3.experiment_pipeline.ddg_pipeline.server_run write-server-plan \
  --processed-csv /path/to/data/processed/dataset_v1.csv \
  --output-root /path/to/server_run \
  --shard-count 4 \
  --methods foldx,rosetta,esm2 \
  --foldx-runs 5 \
  --rosetta-runs 20 \
  --rosetta-top-k 3 \
  --openfold-python python
```

The generated `server_commands.sh` performs the full staged workflow:

1. build a structure-stage CSV with validated sequences
2. split the structure CSV into deterministic shards
3. run OpenFold3 structure prediction per shard
4. merge OpenFold3 structure-stage results
5. build canonical ddG records
6. run ddG shards with incremental `results.csv` writes
7. merge shard outputs into final results

The ddG shard runner writes one row artifact per method/case under `outputs/rows/`, then rewrites `outputs/results.csv` after each completed method. This makes long jobs resumable and prevents losing completed FoldX, Rosetta, or ESM2 results if a later case fails.

## Limitations

- only single-point mutations are supported
- current aggregation is based on FoldX BuildModel energies
- Rosetta values are reported in REU, not kcal/mol
- ESM2 values are reported as pseudo-log-likelihood differences, not kcal/mol
- mixed-method metrics exclude non-kcal/mol rows, so Rosetta rows are not folded into the same numeric benchmark metrics as FoldX
- mixed-method metrics exclude non-kcal/mol rows, so ESM2 rows are also excluded from the same numeric benchmark metrics as FoldX
- shared tabular outputs include an explicit `unit` field because FoldX, Rosetta, and ESM2 are not numerically comparable without it
- metrics are skipped when fewer than 5 successful predictions are available
- validation depends on a FoldX environment that actually writes the repaired PDB artifact

## Why This Protocol Is Scientifically Valid

The stage now supports a first meaningful comparative benchmark across structural and sequence representations:

- FoldX for deterministic structure-based ΔΔG estimation in kcal/mol
- Rosetta ddg_monomer for stochastic structure-based ΔΔG estimation in REU
- ESM2 for sequence-based zero-shot mutation scoring in log-probability units

All three paths keep strict mutation validation, explicit runtime artifacts, and fail-fast behavior. FoldX preserves audited repaired-structure propagation, Rosetta preserves auditable repeated sampling with process-level parallel execution and explicit aggregation, and ESM2 provides a fully local sequence-only baseline without structure-source duplication.
