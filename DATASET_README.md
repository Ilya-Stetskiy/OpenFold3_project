# 🇷🇺 README (RU)

# Подготовка датасета для ddG-экспериментов

## Кратко

Этот этап превращает FireProt-like таблицу мутаций в строгий и воспроизводимый CSV для экспериментов по protein mutation ddG. Основная задача - оставить только те строки, где экспериментальный ddG, sequence position и реальный residue id в PDB-структуре согласованы между собой. Результат готов для последующего сравнения OpenFold и FoldX, но сам не запускает моделирование структур или расчет ddG.

## Обзор

Этап работы с датасетом решает одну задачу: подготовить надежный входной набор мутаций без скрытых ошибок позиционирования.

Он делает это через несколько уровней проверки:

- загружает FireProt-like архив с raw CSV и PDB-структурами;
- оставляет только single-point mutations с экспериментальным ddG;
- нормализует мутацию в формат `A123B`;
- проверяет canonical amino acids;
- проверяет `ddG` в диапазоне `[-10, 10]`;
- трактует `pdb_position` как индекс в упорядоченном списке residues, а не как PDB residue id;
- извлекает реальный `pdb_residue_id` из структуры;
- валидирует совпадение WT residue по `pdb_sequence` и по PDB/mmCIF atom records;
- сохраняет все отброшенные строки с причиной удаления;
- записывает metadata с hash, версией preprocessing и статистикой фильтрации.

Текущая версия preprocessing: `v1.1`.

## Возможности

- Строгая output schema для последующих экспериментов.
- Детерминированные preprocessing и сортировка.
- Проверка cache через metadata и SHA256 hash.
- Корректная интерпретация FireProt `pdb_position` как индекса в ordered residues.
- Сохранение PDB residue identifiers строками, включая значения с insertion codes.
- Полный audit trail для отброшенных строк.
- Проверки sequence, structure, mutation format, ddG range, duplicates и локальных путей.
- Минимальные зависимости: стандартная библиотека Python, `pandas` и существующие OpenFold3 utilities для парсинга структур.

## Ключевые идеи и улучшения

### Корректная семантика позиций

Raw-поле FireProt `pdb_position` не считается настоящим PDB residue identifier. Оно интерпретируется как индекс в упорядоченном списке canonical residues выбранной chain.

Mapping выполняется так:

```text
sequence position -> pdb_position index -> ordered structure residue -> real pdb_residue_id
```

Это защищает от тихих ошибок, когда числовое `pdb_position` случайно совпадает с чужим PDB residue id.

### Две независимые проверки residue

Если в raw row есть `pdb_sequence`, код сначала проверяет:

```text
pdb_sequence[pdb_position_index] == wt_residue
```

Только после этого парсится структура и проверяется:

```text
ordered_chain_residues[pdb_position_index].aa == wt_residue
```

### Полная проверяемость результата

Processed dataset не является единственным артефактом. Каждая отброшенная raw row сохраняется в `discarded_rows.csv` с `discard_reason`, а `dataset_v1_meta.json` фиксирует row counts, filtering stats, версии, hash и runtime environment.

## Общий pipeline

1. Разрешить пути через `DatasetConfig`.
2. Проверить валидность cache для processed dataset.
3. Скачать или переиспользовать FireProt-like архив.
4. Загрузить raw CSV.
5. Нормализовать и проверить raw mutation rows.
6. Проверить `pdb_sequence` относительно `pdb_position` и WT residue.
7. Распарсить соответствующую PDB/mmCIF structure.
8. Построить ordered canonical residues для нужной chain.
9. Разрешить реальный `pdb_residue_id` через индекс `pdb_position`.
10. Проверить identity residue в структуре.
11. Детерминированно отсортировать строки и удалить duplicates.
12. Сохранить processed dataset, discarded rows и metadata.

## Установка

Запускать из корня репозитория:

```bash
cd /mnt/d/Proga/OpenFold_codex/OpenFold3_project
```

Нужны:

- Python 3.12-compatible interpreter;
- `pandas`;
- существующий модуль `openfold-3/openfold3/benchmark/cif_utils.py`;
- network access для первой загрузки архива, если `data/raw/` еще не содержит source archive и extracted CSV.

Для этого этапа не нужны OpenFold inference, FoldX binary, GPU worker, multiprocessing setup или model checkpoints.

## Использование

### Проверить наличие processed dataset

```python
from pipeline.dataset import DatasetConfig, check_dataset_exists

config = DatasetConfig()
print(check_dataset_exists(config))
```

### Скачать raw source files

```python
from pipeline.dataset import DatasetConfig, download_dataset

config = DatasetConfig()
raw_csv_path = download_dataset(config)
print(raw_csv_path)
```

Raw files сохраняются здесь:

```text
data/raw/
```

### Выполнить preprocessing

```python
from pipeline.dataset import DatasetConfig, preprocess_dataset

config = DatasetConfig(force_preprocess=True)
processed_path = preprocess_dataset(config)
print(processed_path)
```

### Загрузить processed dataset

```python
from pipeline.dataset import DatasetConfig, load_dataset

config = DatasetConfig()
df = load_dataset(config)
print(df.dtypes)
print(df.head())
```

`load_dataset()` сохраняет строковую семантику для:

```text
protein_id
pdb_id
chain
pdb_residue_id
pdb_path
```

## Структура проекта

Файлы, относящиеся к dataset stage:

```text
pipeline/dataset.py
notebooks/main_experiment.ipynb
data/raw/
data/processed/dataset_v1.csv
data/processed/discarded_rows.csv
data/processed/dataset_v1_meta.json
```

Ответственность файлов:

- `pipeline/dataset.py` содержит configuration, download, preprocessing, validation, cache checks, sorting, metadata writing и loading.
- `notebooks/main_experiment.ipynb` сейчас содержит только CONFIG и DATASET cells.
- `data/raw/` хранит скачанный FireProt-like archive и extracted raw files.
- `data/processed/dataset_v1.csv` - строгий processed dataset.
- `data/processed/discarded_rows.csv` - audit trail для удаленных строк.
- `data/processed/dataset_v1_meta.json` - metadata для воспроизводимости.

## Технические детали

### Processed schema

Processed dataset содержит колонки:

```text
protein_id
pdb_id
chain
position
wt_residue
mut_residue
experimental_ddg
mutation_id
pdb_residue_id
pdb_path
```

Смысл колонок:

- `protein_id`: protein-level identifier, обычно UniProt id из raw source.
- `pdb_id`: нормализованный четырехсимвольный PDB id.
- `chain`: нормализованный chain id через `strip().upper()`.
- `position`: sequence position из raw dataset, 1-based.
- `wt_residue`: wild-type residue, one-letter canonical amino acid.
- `mut_residue`: mutant residue, one-letter canonical amino acid.
- `experimental_ddg`: experimental ddG после range filtering.
- `mutation_id`: нормализованная mutation label, например `A123B`.
- `pdb_residue_id`: реальный residue id из parsed structure, сохраненный строкой.
- `pdb_path`: локальный путь к PDB/mmCIF file, использованному для validation.

### Правила фильтрации

Строка остается только если выполняются все условия:

- валидный PDB id;
- валидные chain, WT residue, mutant residue и mutation position;
- только canonical amino acids;
- `experimental_ddg` присутствует и находится в диапазоне `[-10, 10]`;
- `pdb_position` является неотрицательным целым индексом;
- если есть `pdb_sequence`, то `pdb_sequence[pdb_position] == wt_residue`;
- structure file существует или может быть скачан;
- ordered structure residue на позиции `pdb_position` существует;
- structure residue identity совпадает с `wt_residue`;
- нет duplicate `(protein_id, mutation_id)` после deterministic sorting;
- нет duplicate `(pdb_id, chain, pdb_residue_id, mut_residue)`.

### Причины удаления строк

`discarded_rows.csv` использует такие `discard_reason`:

```text
invalid_pdb_id
invalid_mutation
invalid_ddg
sequence_structure_mismatch
mapping_failed
duplicate
```

### Текущий generated dataset

Для текущих generated files:

```text
rows: 3113
proteins: 88
ddG range: -8.64 .. 9.8
discarded rows: 1884
invalid_ddg: 1561
mapping_failed: 303
duplicate: 20
sequence_structure_mismatch: 0
```

### Cache validation

`check_dataset_exists()` возвращает `True` только если:

- существует `dataset_v1.csv`;
- существует `dataset_v1_meta.json`;
- output columns совпадают с ожидаемой schema;
- metadata preprocessing version равна `v1.1`;
- metadata columns совпадают с текущей schema;
- metadata `dataset_hash` совпадает с текущим CSV SHA256;
- metadata row count совпадает с количеством строк в CSV.

Если хотя бы одна проверка не проходит, preprocessing выполняется заново.

## Ограничения

- Этот этап поддерживает текущий FireProt-like archive source и его source schema.
- Он не запускает OpenFold, FoldX, Rosetta, Prompt-DDG или GPU worker.
- Он не выравнивает произвольные external protein sequences на structures; он использует source-provided `pdb_position` index и строго валидирует его.
- Строки с отсутствующим или inconsistent structure mapping удаляются, а не исправляются автоматически.
- Metadata содержит preprocessing timestamp, поэтому metadata file меняется между forced reruns даже при идентичном CSV content.

## Будущая работа

- Добавить tests для mapping edge cases: insertion codes, missing residues, shifted numbering и non-contiguous residue ids.
- Добавить поддержку дополнительных dataset sources через explicit source adapters.
- Добавить небольшой fixture dataset для быстрой CI validation.
- Добавить optional sequence-to-structure alignment для источников без `pdb_position`.
- Задокументировать, как downstream OpenFold и FoldX stages должны по-разному использовать `position` и `pdb_residue_id`.


---

# 🇬🇧 README (EN)

# Dataset Stage for ddG Experiments

## TL;DR

This dataset stage converts a FireProt-like mutation table into a strict, reproducible CSV for protein mutation ddG experiments. Its main purpose is to keep only rows where experimental ddG, sequence position, and the actual residue id in the structure are mutually consistent. The output is ready for later OpenFold and FoldX comparison stages, but this stage does not run structure prediction or ddG scoring.

## Overview

The dataset stage has one responsibility: prepare a reliable mutation dataset without hidden residue-mapping errors.

It does this through layered validation:

- downloads a FireProt-like archive with raw CSV and PDB structures;
- keeps only single-point mutations with experimental ddG;
- normalizes mutation labels to `A123B`;
- validates canonical amino acids;
- filters `ddG` to the `[-10, 10]` range;
- treats `pdb_position` as an index into ordered structure residues, not as a PDB residue id;
- resolves the actual `pdb_residue_id` from the structure;
- validates WT residue identity against both `pdb_sequence` and parsed structure records;
- stores every discarded row with a discard reason;
- writes metadata with a hash, preprocessing version, and filtering statistics.

Current preprocessing version: `v1.1`.

## Features

- Strict output schema for downstream experiments.
- Deterministic preprocessing and sorting.
- Strong cache validation using metadata and SHA256 hash.
- Correct FireProt `pdb_position` interpretation as an ordered-residue index.
- PDB residue identifiers are stored as strings, preserving compatibility with insertion codes.
- Complete discarded-row audit trail.
- Built-in checks for sequence, structure, mutation format, ddG range, duplicates, and local paths.
- Minimal dependencies: Python standard library plus `pandas` and existing OpenFold3 structure parsing utilities.

## Key Ideas / Improvements

### Correct position semantics

The raw FireProt field `pdb_position` is not treated as a real PDB residue identifier. It is interpreted as an index into the ordered canonical residues of the selected chain.

The mapping is:

```text
sequence position -> pdb_position index -> ordered structure residue -> real pdb_residue_id
```

This prevents silent errors where a numeric `pdb_position` accidentally matches an unrelated PDB residue id.

### Two independent residue checks

When `pdb_sequence` is present, the code first checks:

```text
pdb_sequence[pdb_position_index] == wt_residue
```

Only then does it parse the structure and check:

```text
ordered_chain_residues[pdb_position_index].aa == wt_residue
```

### Auditable output

The processed dataset is not the only artifact. Every discarded raw row is preserved in `discarded_rows.csv` with `discard_reason`, and `dataset_v1_meta.json` records row counts, filtering statistics, versions, hash, and runtime environment information.

## High-level Pipeline

1. Resolve `DatasetConfig` paths.
2. Check whether the processed dataset cache is valid.
3. Download or reuse the FireProt-like archive.
4. Load the raw CSV.
5. Normalize and validate raw mutation rows.
6. Check `pdb_sequence` against `pdb_position` and WT residue.
7. Parse the matching PDB/mmCIF structure.
8. Build ordered canonical residues for the requested chain.
9. Resolve the real `pdb_residue_id` from the `pdb_position` index.
10. Validate structure residue identity.
11. Sort deterministically and remove duplicates.
12. Save the processed dataset, discarded rows, and metadata.

## Installation

Run from the repository root:

```bash
cd /mnt/d/Proga/OpenFold_codex/OpenFold3_project
```

Required runtime pieces:

- Python 3.12-compatible interpreter;
- `pandas`;
- existing repository module `openfold-3/openfold3/benchmark/cif_utils.py`;
- network access for the first archive download, unless `data/raw/` already contains the source archive and extracted CSV.

This stage does not require OpenFold inference, a FoldX binary, GPU worker setup, multiprocessing, or model checkpoints.

## Usage

### Check whether the processed dataset exists

```python
from pipeline.dataset import DatasetConfig, check_dataset_exists

config = DatasetConfig()
print(check_dataset_exists(config))
```

### Download raw source files

```python
from pipeline.dataset import DatasetConfig, download_dataset

config = DatasetConfig()
raw_csv_path = download_dataset(config)
print(raw_csv_path)
```

Raw files are stored under:

```text
data/raw/
```

### Preprocess the dataset

```python
from pipeline.dataset import DatasetConfig, preprocess_dataset

config = DatasetConfig(force_preprocess=True)
processed_path = preprocess_dataset(config)
print(processed_path)
```

### Load the processed dataset

```python
from pipeline.dataset import DatasetConfig, load_dataset

config = DatasetConfig()
df = load_dataset(config)
print(df.dtypes)
print(df.head())
```

`load_dataset()` preserves string semantics for:

```text
protein_id
pdb_id
chain
pdb_residue_id
pdb_path
```

## Project Structure

Dataset-specific files:

```text
pipeline/dataset.py
notebooks/main_experiment.ipynb
data/raw/
data/processed/dataset_v1.csv
data/processed/discarded_rows.csv
data/processed/dataset_v1_meta.json
```

Responsibilities:

- `pipeline/dataset.py` contains dataset configuration, download, preprocessing, validation, cache checks, sorting, metadata writing, and loading.
- `notebooks/main_experiment.ipynb` currently contains only CONFIG and DATASET cells.
- `data/raw/` stores the downloaded FireProt-like archive and extracted raw files.
- `data/processed/dataset_v1.csv` is the strict processed dataset.
- `data/processed/discarded_rows.csv` is the audit trail for removed rows.
- `data/processed/dataset_v1_meta.json` records reproducibility metadata.

## Technical Details

### Processed schema

The processed dataset columns are:

```text
protein_id
pdb_id
chain
position
wt_residue
mut_residue
experimental_ddg
mutation_id
pdb_residue_id
pdb_path
```

Column meanings:

- `protein_id`: protein-level identifier, usually the UniProt id from the raw source.
- `pdb_id`: normalized four-character PDB id.
- `chain`: normalized chain id, `strip().upper()`.
- `position`: sequence position from the raw dataset, 1-based.
- `wt_residue`: wild-type residue, one-letter canonical amino acid.
- `mut_residue`: mutant residue, one-letter canonical amino acid.
- `experimental_ddg`: experimental ddG value after range filtering.
- `mutation_id`: normalized mutation label, such as `A123B`.
- `pdb_residue_id`: real residue id from the parsed structure, stored as a string.
- `pdb_path`: local path to the PDB/mmCIF file used for validation.

### Filtering rules

Rows are kept only if all conditions pass:

- valid PDB id;
- valid chain, WT residue, mutant residue, and mutation position;
- canonical amino acids only;
- `experimental_ddg` is present and within `[-10, 10]`;
- `pdb_position` is a non-negative integer index;
- if `pdb_sequence` exists, `pdb_sequence[pdb_position] == wt_residue`;
- structure file exists or can be downloaded;
- ordered structure residue at `pdb_position` exists;
- structure residue identity matches `wt_residue`;
- no duplicate `(protein_id, mutation_id)` after deterministic sorting;
- no duplicate `(pdb_id, chain, pdb_residue_id, mut_residue)`.

### Discard reasons

`discarded_rows.csv` uses these reasons:

```text
invalid_pdb_id
invalid_mutation
invalid_ddg
sequence_structure_mismatch
mapping_failed
duplicate
```

### Current generated dataset

For the current generated files:

```text
rows: 3113
proteins: 88
ddG range: -8.64 .. 9.8
discarded rows: 1884
invalid_ddg: 1561
mapping_failed: 303
duplicate: 20
sequence_structure_mismatch: 0
```

### Cache validation

`check_dataset_exists()` returns `True` only when:

- `dataset_v1.csv` exists;
- `dataset_v1_meta.json` exists;
- output columns match the expected schema;
- metadata preprocessing version is `v1.1`;
- metadata columns match the current schema;
- metadata `dataset_hash` matches the current CSV SHA256;
- metadata row count matches the CSV line count.

If any check fails, preprocessing is recomputed.

## Limitations

- This stage supports the current FireProt-like archive source and source schema.
- It does not run OpenFold, FoldX, Rosetta, Prompt-DDG, or any GPU worker.
- It does not align arbitrary external protein sequences to structures; it relies on the source-provided `pdb_position` index and validates it strictly.
- Rows with missing or inconsistent structure mapping are discarded rather than repaired.
- Metadata includes a preprocessing timestamp, so the metadata file changes between forced reruns even when the CSV content is identical.

## Future Work

- Add tests for mapping edge cases: insertion codes, missing residues, shifted numbering, and non-contiguous residue ids.
- Add support for additional dataset sources with explicit source adapters.
- Add a small fixture dataset for fast CI validation.
- Add optional sequence-to-structure alignment for sources that do not provide `pdb_position`.
- Document how downstream OpenFold and FoldX stages should consume `position` and `pdb_residue_id` differently.

