# 🇷🇺 README (RU)

# Structure Stage

## TL;DR
`structure stage` — это воспроизводимый orchestration-слой для подготовки кейсов точечных мутаций и запуска бэкендов моделирования структур. Он не меняет core OpenFold3 и не реализует собственную биофизику: его задача — стабильно превратить dataset в набор case-директорий, подготовить входы для FoldX и OpenFold3, закэшировать результаты и собрать единый `results.csv`.

## Overview
- Принимает CSV-датасет с уже нормализованными мутациями и исходными последовательностями.
- Создаёт детерминированную файловую структуру для каждого mutation case.
- Поддерживает два независимых backend-а: `foldx` и `openfold3`.
- Сохраняет `manifest.json` на case-уровне и повторно использует только валидные артефакты.
- Разделяет orchestration errors и runtime failures backend-ов.
- Умеет работать в `dry-run` режиме без реального моделирования.

## Features
- Типизированные dataclass-модели для mutation case и backend result.
- Подготовка `case.json` и `mutant.fasta` для каждого кейса.
- Dry-run для проверки dataset wiring и файловой структуры.
- FoldX backend через отдельный orchestration wrapper.
- OpenFold3 backend через `python -m openfold3.run_openfold predict`.
- Cache по `config_hash` с валидацией наличия итогового артефакта.
- Агрегация результатов в `results.csv`.
- CLI smoke/e2e покрытие для базового pipeline path.

## Key Ideas / Improvements
- **Минимальная интеграция:** stage использует внешние инструменты как black-box backend-ы и не вторгается в `openfold3.core`.
- **Детерминированный input contract:** каждый case имеет один и тот же набор входных файлов и стабильный `case_id`.
- **Явный cache:** reuse разрешён только для `status=success`, совпавшего `config_hash` и существующего artifact path.
- **Fail isolation:** падение одного backend-а не должно ломать другой backend или весь batch.
- **Прозрачная summary-модель:** case-level и backend-level статистика считаются отдельно.

## High-level Pipeline
1. CLI читает dataset CSV через `pandas`.
2. Каждая строка конвертируется в `MutationCase`.
3. Для каждого case применяется `apply_mutation()` к исходной последовательности.
4. В `output_dir/cases/{case_id}/input/` создаются `case.json` и `mutant.fasta`.
5. Для каждого backend-а проверяется cache по `manifest.json`.
6. Если cache невалиден, backend запускается заново.
7. Результат backend-а записывается в `manifest.json`.
8. После batch-run формируется общий `results.csv`.

## Installation

### Requirements
- Python 3.11+ в окружении проекта.
- `pandas`.
- Для `foldx` backend: установленный и исполняемый FoldX binary.
- Для `openfold3` backend: рабочий запуск `python -m openfold3.run_openfold predict`.

### Repo context
README относится к каталогу:
`openfold-3/openfold3/experiment_pipeline/structure/`

Если нужен только dry-run или проверка структуры кейсов, достаточно Python-зависимостей проекта.

## Usage

### 1. Dry-run
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset openfold3/tests/test_data/experiment_pipeline/structure/test.csv   --output-dir /tmp/of3-structure-dry-run   --dry-run   --backend both
```

Что делает dry-run:
- валидирует dataset contract
- создаёт case directories
- пишет `case.json` и `mutant.fasta`
- не запускает FoldX и OpenFold3

### 2. FoldX only
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend foldx   --foldx-binary /path/to/foldx
```

### 3. OpenFold3 only
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend openfold3   --openfold-python python3   --openfold-runner-yaml path/to/runner.yaml   --openfold-inference-ckpt-path path/to/checkpoint.pt
```

### 4. Both backends
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend both   --foldx-binary /path/to/foldx   --openfold-python python3
```

### 5. Strict config hash
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend both   --strict-config-hash
```

Это включает content hash для checkpoint-файла, а не только metadata fingerprint.

## Project Structure
- `models.py` — dataclass-контракт для `MutationCase`, `BackendResult`, `CaseManifest`.
- `sequence.py` — применение одиночной мутации к аминокислотной последовательности.
- `runner.py` — orchestration runner, cache decision, manifest update, `results.csv`.
- `cache.py` — валидация cache hit по `status`, `config_hash` и существованию итогового artifact.
- `manifest.py` — загрузка/сохранение `manifest.json` и сериализация backend state.
- `cli.py` — CLI, config hashing, dataset loading, summary aggregation.
- `backends/foldx_backend.py` — запуск FoldX BuildModel и нормализация `mutant.pdb`.
- `backends/openfold_backend.py` — запуск `run_openfold predict` и выбор итогового `model.cif`.

## Technical Details

### Dataset contract
Ожидается CSV с колонками, достаточными для `MutationCase.from_row(...)`:
- `protein_id`
- `pdb_id`
- `chain`
- `position`
- `wt_residue`
- `mut_residue`
- `experimental_ddg`
- `mutation_id`
- `pdb_residue_id`
- `pdb_path`
- `sequence`

`sequence` обязателен именно для structure stage, потому что stage строит `mutant_sequence` через `apply_mutation()`.

### Mutation application
`apply_mutation(sequence, position, wt, mut)`:
- использует 1-based индексацию
- валидирует границы позиции
- валидирует совпадение WT residue
- возвращает мутантную последовательность

### Case layout
Для каждого кейса создаётся структура:
```text
output_dir/
  cases/
    {case_id}/
      input/
        case.json
        mutant.fasta
      manifest.json
      foldx/
      openfold3/
```

### Manifest format
`manifest.json` хранит состояние backend-ов по ключам `foldx` и `openfold3`:
- `status`
- `config_hash`
- `output_dir`
- `artifact_paths`
- `structure_path`
- `message`

### Cache policy
Backend считается cache-hit только если:
- `status == "success"`
- `config_hash` совпадает
- `structure_path` существует в файловой системе

Если артефакт удалён, cache инвалидируется, и backend будет перезапущен.

### Config hash
`config_hash` строится через `sha256(json.dumps(..., sort_keys=True, ensure_ascii=False))` и включает:
- выбранные backend-ы
- `dry_run`
- `strict_config_hash`
- fingerprint FoldX binary
- Python executable для OpenFold3
- fingerprint `runner_yaml`
- fingerprint checkpoint path

Для PATH-style binary, например `foldx`, сначала выполняется `which`-резолвинг, чтобы cache зависел от реального бинарника, а не от строкового alias.

### Backend dispatch
`StructureRunner` поддерживает две сигнатуры `backend.run(...)`:
- `(case, case_dir, config_hash)`
- `(case, case_dir, config_hash, mutated_sequence)`

Все остальные сигнатуры считаются orchestration contract error и поднимаются как `BackendContractError`.

### FoldX backend
FoldX backend:
- копирует исходную структуру в case-local directory
- создаёт `individual_list.txt`
- чистит `foldx/output/` перед новым запуском
- вызывает binary без `shell=True`
- ожидает результат текущего запуска и сохраняет его как `foldx/output/mutant.pdb`

### OpenFold3 backend
OpenFold3 backend:
- пишет `wt.fasta`
- пишет минимальный `query.json`
- вызывает `python -m openfold3.run_openfold predict`
- чистит `openfold3/output/` перед rerun
- ищет итоговый CIF среди `*.cif`
- не использует случайный `sorted()[0]`
- принимает единственный candidate или единственный `final/model` candidate
- сохраняет итоговый файл как `openfold3/output/model.cif`

### Results aggregation
После non-dry-run batch-а создаётся `results.csv` с колонками:
- `case_id`
- `openfold3_status`
- `foldx_status`
- `openfold3_path`
- `foldx_path`

### Summary reporting
CLI печатает два блока:
- `CASE SUMMARY`
- `BACKEND SUMMARY`

Сейчас учитываются статусы:
- `ok`
- `failed`
- `cached`
- `unavailable`

## Limitations
- Stage поддерживает только одиночные аминокислотные замены на уровне sequence string.
- Dataset должен уже содержать корректный `sequence` и `pdb_path`; stage не строит их сам.
- OpenFold3 backend сейчас формирует минимальный single-chain query и не управляет расширенными MSA/template сценариями вне доступных CLI-параметров.
- Cache валидирует только наличие итогового артефакта, но не проверяет checksum или содержимое файла.
- `results.csv` отражает итоговые backend status/path, но не хранит полную историю повторных прогонов.

## Future Work
- Добавить отдельный adapter для richer OpenFold query generation.
- Добавить более строгую валидацию manifest schema и artifact compatibility.
- Расширить summary до machine-readable report JSON верхнего уровня.
- Добавить отдельные integration tests для реального OpenFold3 backend path.
- Добавить слой downstream-аналитики для сравнения FoldX/OpenFold3 структурных результатов.

---

# 🇬🇧 README (EN)

# Structure Stage

## TL;DR
The `structure stage` is a reproducible orchestration layer that prepares single-mutation cases and runs structure-modeling backends. It does not modify OpenFold3 core code or implement its own modeling logic; its job is to turn a dataset into deterministic case directories, prepare FoldX/OpenFold3 inputs, cache valid outputs, and emit a single `results.csv`.

## Overview
- Reads a CSV dataset with normalized mutations and source sequences.
- Creates a deterministic on-disk layout for every mutation case.
- Supports two independent backends: `foldx` and `openfold3`.
- Stores per-case `manifest.json` files and reuses only validated cached artifacts.
- Separates orchestration contract errors from backend runtime failures.
- Supports `dry-run` mode for wiring and filesystem checks without real modeling.

## Features
- Typed dataclass models for mutation cases and backend results.
- Automatic generation of `case.json` and `mutant.fasta` for each case.
- Dry-run mode for validating dataset wiring and case layout.
- FoldX backend through a dedicated orchestration wrapper.
- OpenFold3 backend through `python -m openfold3.run_openfold predict`.
- Cache based on `config_hash` plus artifact existence validation.
- Aggregated `results.csv` output.
- CLI smoke and end-to-end coverage for the main pipeline path.

## Key Ideas / Improvements
- **Minimal integration:** the stage treats external tools as black-box backends and does not patch `openfold3.core`.
- **Deterministic input contract:** every case gets the same input files and a stable `case_id`.
- **Explicit cache discipline:** reuse is allowed only for `status=success`, matching `config_hash`, and an existing artifact path.
- **Failure isolation:** one backend failure should not take down the other backend or the whole batch.
- **Transparent summaries:** case-level and backend-level accounting are computed separately.

## High-level Pipeline
1. The CLI reads the dataset CSV with `pandas`.
2. Each row is converted into a `MutationCase`.
3. `apply_mutation()` is used to build the mutant sequence.
4. `case.json` and `mutant.fasta` are written to `output_dir/cases/{case_id}/input/`.
5. Each backend is checked against `manifest.json` for cache reuse.
6. If cache is invalid, the backend is executed again.
7. Backend results are stored back into `manifest.json`.
8. A batch-level `results.csv` is generated after the run.

## Features
- Typed dataclass-based case contract.
- Stable directory layout for dry-run and real execution.
- FoldX and OpenFold3 backend selection through CLI.
- Config-aware cache invalidation.
- Per-case manifests and batch-level result aggregation.

## Installation

### Requirements
- Python 3.11+ in the project environment.
- `pandas`.
- For the `foldx` backend: an installed and executable FoldX binary.
- For the `openfold3` backend: a working `python -m openfold3.run_openfold predict` setup.

### Repository context
This README documents:
`openfold-3/openfold3/experiment_pipeline/structure/`

If you only need dry-run validation or filesystem preparation, the normal Python project dependencies are enough.

## Usage

### 1. Dry-run
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset openfold3/tests/test_data/experiment_pipeline/structure/test.csv   --output-dir /tmp/of3-structure-dry-run   --dry-run   --backend both
```

Dry-run behavior:
- validates the dataset contract
- creates case directories
- writes `case.json` and `mutant.fasta`
- does not run FoldX or OpenFold3

### 2. FoldX only
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend foldx   --foldx-binary /path/to/foldx
```

### 3. OpenFold3 only
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend openfold3   --openfold-python python3   --openfold-runner-yaml path/to/runner.yaml   --openfold-inference-ckpt-path path/to/checkpoint.pt
```

### 4. Both backends
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend both   --foldx-binary /path/to/foldx   --openfold-python python3
```

### 5. Strict config hash
```bash
python -m openfold3.experiment_pipeline.structure.cli   --dataset path/to/dataset.csv   --output-dir path/to/output   --backend both   --strict-config-hash
```

This switches checkpoint hashing from metadata-only to full file-content hashing.

## Project Structure
- `models.py` — dataclass contracts for `MutationCase`, `BackendResult`, and `CaseManifest`.
- `sequence.py` — single-residue mutation logic for protein sequences.
- `runner.py` — orchestration runner, cache decisions, manifest updates, and `results.csv` generation.
- `cache.py` — cache validation based on `status`, `config_hash`, and final artifact existence.
- `manifest.py` — `manifest.json` loading/saving and backend state serialization.
- `cli.py` — CLI entry point, config hashing, dataset loading, and summary aggregation.
- `backends/foldx_backend.py` — FoldX BuildModel execution and normalized `mutant.pdb` output.
- `backends/openfold_backend.py` — `run_openfold predict` execution and `model.cif` selection.

## Technical Details

### Dataset contract
The stage expects a CSV with the fields required by `MutationCase.from_row(...)`:
- `protein_id`
- `pdb_id`
- `chain`
- `position`
- `wt_residue`
- `mut_residue`
- `experimental_ddg`
- `mutation_id`
- `pdb_residue_id`
- `pdb_path`
- `sequence`

`sequence` is mandatory for the structure stage because mutant sequences are built through `apply_mutation()`.

### Mutation application
`apply_mutation(sequence, position, wt, mut)`:
- uses 1-based indexing
- validates sequence bounds
- validates the observed wild-type residue
- returns the mutant sequence

### Case layout
Each case is written as:
```text
output_dir/
  cases/
    {case_id}/
      input/
        case.json
        mutant.fasta
      manifest.json
      foldx/
      openfold3/
```

### Manifest format
`manifest.json` stores backend state under `foldx` and `openfold3` keys:
- `status`
- `config_hash`
- `output_dir`
- `artifact_paths`
- `structure_path`
- `message`

### Cache policy
A backend is treated as cached only if:
- `status == "success"`
- `config_hash` matches
- `structure_path` exists on disk

If the artifact is missing, cache is invalidated and the backend is rerun.

### Config hash
`config_hash` is built with `sha256(json.dumps(..., sort_keys=True, ensure_ascii=False))` and includes:
- selected backends
- `dry_run`
- `strict_config_hash`
- FoldX binary fingerprint
- OpenFold Python executable
- `runner_yaml` fingerprint
- checkpoint fingerprint

For PATH-style binaries such as `foldx`, the CLI resolves the actual executable through `which` before hashing, so cache reuse depends on the real binary, not only the alias string.

### Backend dispatch
`StructureRunner` supports two `backend.run(...)` signatures:
- `(case, case_dir, config_hash)`
- `(case, case_dir, config_hash, mutated_sequence)`

Any other signature is treated as an orchestration contract error and raised as `BackendContractError`.

### FoldX backend
The FoldX backend:
- copies the source structure into the case-local directory
- creates `individual_list.txt`
- cleans `foldx/output/` before reruns
- calls the binary without `shell=True`
- resolves only current-run outputs and stores the normalized result as `foldx/output/mutant.pdb`

### OpenFold3 backend
The OpenFold3 backend:
- writes `wt.fasta`
- writes a minimal `query.json`
- runs `python -m openfold3.run_openfold predict`
- cleans `openfold3/output/` before reruns
- searches for final CIF candidates inside the output tree
- does not use a blind `sorted()[0]` fallback
- accepts either a single candidate or a single preferred `final/model` candidate
- stores the normalized output as `openfold3/output/model.cif`

### Results aggregation
After a non-dry-run batch, `results.csv` is created with:
- `case_id`
- `openfold3_status`
- `foldx_status`
- `openfold3_path`
- `foldx_path`

### Summary reporting
The CLI prints two blocks:
- `CASE SUMMARY`
- `BACKEND SUMMARY`

The current summary model explicitly tracks:
- `ok`
- `failed`
- `cached`
- `unavailable`

## Limitations
- The stage currently supports only single amino-acid substitutions at the sequence-string level.
- The dataset must already provide valid `sequence` and `pdb_path` values; the stage does not derive them.
- The OpenFold3 backend currently builds a minimal single-chain query and does not expose richer MSA/template workflows beyond the CLI flags that already exist.
- Cache validation checks artifact existence, not file checksum or semantic correctness.
- `results.csv` stores the latest backend status/path view, not the full rerun history.

## Future Work
- Add a dedicated adapter for richer OpenFold query generation.
- Add stricter manifest schema validation and backend compatibility checks.
- Add a machine-readable top-level JSON summary in addition to CLI text output.
- Add dedicated integration coverage for real OpenFold3 backend execution.
- Add downstream analysis utilities for comparing FoldX and OpenFold3 structural outputs.
