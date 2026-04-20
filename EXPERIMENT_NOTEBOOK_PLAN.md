# План experiment notebook pipeline

## Цель ветки

Ветка `feature/experiment-notebook` предназначена для минимальной подготовки будущего notebook-based pipeline для моделирования структур mutant protein и оценки ddG. На этом шаге не реализуется полный pipeline, multiprocessing, worker queues или полноценный notebook.

## Найденные переиспользуемые модули

### Dataset и входные структуры

- `openfold-3/openfold3/core/data/framework/single_datasets/*` - базовые dataset-компоненты OpenFold3; полезны как reference для внутренних форматов, но не как прямой пользовательский dataset loader для notebook.
- `openfold-3/openfold3/core/data/io/dataset_cache.py` - работа с dataset cache.
- `openfold-3/openfold3/benchmark/models.py` - dataclass-модели `MutationInput`, `BenchmarkCase`, `StructureSummary`.
- `openfold-3/openfold3/benchmark/structure_source.py` - разрешение `pdb_id`/локальной структуры, скачивание mmCIF, валидация mutation site, извлечение protein sequence.
- `openfold_notebooks/openfold3_length_benchmark/composition.py` - получение состава PDB entry и преобразование в molecules для notebook.

### Mutation processing

- `openfold_notebooks/helpers/of_notebook_lib/query_builders.py` - `normalize_molecules`, `apply_point_mutation`, `apply_mutation_to_molecules`, `build_single_query_payload`, `build_mutation_scan_payload`.
- `openfold-3/openfold3/mutation_runner.py` - `MutationSpec`, `ScreeningJob`, `MutationScreeningRunner`, cache/reuse MSA/template artifacts, batch mutation screening. Для первого notebook pipeline использовать только как готовый backend, не копировать.
- `gpu_worker/runner.py` - `normalize_molecules`, `apply_point_mutation_to_molecules`, worker-oriented запуск OpenFold3. Нужен только если pipeline позже будет подключаться к worker service.

### Structure generation with OpenFold3

- `openfold_notebooks/helpers/of_notebook_lib/config.py` - `RuntimeConfig` для путей и runtime environment.
- `openfold_notebooks/helpers/of_notebook_lib/runner.py` - `run_prediction`, `RunResult`, сбор артефактов OpenFold3 output.
- `openfold_notebooks/helpers/of_notebook_lib/workflows.py` - готовые notebook workflows для single case, mutation scan и server smoke.
- `openfold-3/openfold3/run_openfold.py` и `openfold-3/openfold3/projects/of3_all_atom/runner.py` - нижний уровень запуска OpenFold3.
- `gpu_worker/client.py`, `gpu_worker/runner.py`, `gpu_worker/schemas.py` - reusable worker API, отложить до этапа распределенного запуска.

### FoldX и ddG logic

- `openfold_notebooks/helpers/of_notebook_lib/ddg_tools.py` - поиск FoldX/Rosetta binaries, export env, tool status для notebook.
- `openfold_notebooks/helpers/of_notebook_lib/ddg_panel.py` - helper-функции panel preview, FoldX context, visualization rows.
- `openfold-3/openfold3/benchmark/foldx_panel.py` - `build_foldx_panel_mutations`, `run_foldx_panel`, summary/ranking CSV для FoldX panel.
- `openfold-3/openfold3/benchmark/local_edit.py` - local mutation modeling case вокруг FoldX/ddG harness.
- `openfold-3/openfold3/benchmark/methods.py` - `FoldXBuildModelMethod`, `RosettaScoreMethod`, `Saambe3DMethod`, `HeliXonBindingDdgMethod`, `PromptDdgMethod`, `default_methods`, `multiscale_methods`.
- `openfold-3/openfold3/benchmark/harness.py` - `DdgBenchmarkHarness`, `HarnessReport`, `MethodResult`.
- `foldx_worker/cli.py` - отдельный FoldX worker CLI; переиспользовать позже только если нужен isolated FoldX execution.

### Metrics и evaluation

- `openfold_notebooks/helpers/of_notebook_lib/analysis.py` - сбор OpenFold3 samples, dataframe, ranking лучших samples.
- `openfold-3/openfold3/testbench/evaluation.py` - `evaluate_reports`, Pearson/Spearman/Kendall, MAE/RMSE, sign accuracy, top-k overlap.
- `openfold-3/openfold3/core/metrics/*` - confidence/model selection/sample ranking metrics OpenFold3.
- `openfold-3/openfold3/benchmark/mutation_analysis.py` - batch mutation analysis и delta vs WT по методам; файл сейчас есть как untracked в рабочем дереве, поэтому использовать осторожно до фиксации его происхождения.

### Notebook helpers

- `openfold_notebooks/helpers/of_notebook_lib/*` - основной существующий слой notebook helpers.
- `openfold_notebooks/OpenFold3_DDG_Stand.ipynb` и `openfold_notebooks/OpenFold3_FoldX_DDG_Panel.ipynb` - существующие notebooks как reference, не копировать в новую структуру.
- `openfold_notebooks/openfold3_length_benchmark/*` и `openfold_notebooks/openfold3_runtime_benchmark/*` - примеры структуры notebook-oriented Python package.

## Недостающие компоненты

- Явный loader для пользовательской таблицы мутаций с колонками dataset/target/chain/position/wt/mut/experimental_ddg.
- Единый lightweight API в `pipeline/`, который связывает dataset rows, OpenFold3 structure generation, FoldX/ddG evaluation и итоговые metrics.
- Контракт выходных директорий для будущего notebook: где лежат raw inputs, OpenFold3 outputs, FoldX reports, merged metrics и финальные tables.
- Проверка соответствия sequence position и structure residue id для разных PDB/mmCIF numbering schemes.
- Минимальные тестовые fixtures для будущих функций `pipeline/`, когда появится реализация.

## Файлы из соседних веток

На этом шаге перенос файлов из соседних веток не требуется.

Проверенные ветки и вывод:

- `codex/unified-mutation-analysis` и `codex/local-mutation-stand` содержат те же ключевые области: `openfold3/mutation_runner.py`, `openfold3/benchmark/*`, `openfold_notebooks/*`.
- `origin/codex/mutation-screening` содержит mutation screening и data pipeline references, но не дает отдельного минимального scaffold для текущей задачи.
- Уникальный `notebooks/`, `pipeline/` или `EXPERIMENT_NOTEBOOK_PLAN.md` scaffold в соседнем clean checkout не найден.

Если позже выяснится, что нужная функция существует только в соседней ветке, ее следует переносить точечно через `git show <branch>:<path>` после сравнения с текущим файлом, а не копировать крупные блоки.

## Созданный минимальный scaffold

- `notebooks/.gitkeep` - фиксирует будущую директорию для notebook.
- `pipeline/__init__.py` - минимальный пакет будущего pipeline.
- `pipeline/dataset.py` - placeholder для loader/normalizer экспериментального dataset.
- `pipeline/structure.py` - placeholder для OpenFold3 structure generation wrappers.
- `pipeline/ddg.py` - placeholder для FoldX/ddG integration.
- `pipeline/metrics.py` - placeholder для evaluation metrics.
- `pipeline/utils.py` - placeholder для небольших общих утилит.

## Предлагаемая финальная структура notebook

Будущий notebook в `notebooks/` должен быть построен по этапам:

1. Environment check: repo paths, Python environment, OpenFold3/FoldX availability.
2. Dataset load: чтение таблицы мутаций, валидация колонок, нормализация mutation IDs.
3. Target preparation: разрешение `pdb_id` или локальных structures, проверка chain/position/wt residue.
4. OpenFold3 structure generation: запуск WT/mutant queries через существующий runner или `MutationScreeningRunner`.
5. FoldX/ddG evaluation: запуск `run_foldx_panel` или `DdgBenchmarkHarness` на выбранных structures.
6. Metrics merge: объединение predictions, ddG reports и experimental ddG.
7. Evaluation: корреляции, MAE/RMSE, sign accuracy, ranking tables.
8. Export: сохранение clean CSV/JSON summary и короткого markdown report.

## Риски и неоднозначности

- В рабочем дереве уже были незакоммиченные изменения до создания этой ветки. Они не относятся к данному scaffold и не должны смешиваться с будущим коммитом этой задачи.
- `openfold-3/openfold3/benchmark/mutation_analysis.py` сейчас untracked в рабочем дереве; до коммита или подтверждения происхождения его нельзя считать стабильной зависимостью.
- Нужно отдельно зафиксировать формат входной экспериментальной таблицы и правила residue numbering: sequence position против PDB residue id.
- FoldX mutagenesis использует structure residue ids; OpenFold3 mutation payload использует sequence positions. Этот mapping должен быть явной частью будущей реализации.
- Неясно, будет ли первый pipeline запускать OpenFold3 локально, через `gpu_worker`, или через существующий batch mutation runner. Сейчас это намеренно отложено.

## Точный следующий шаг

Реализовать только `pipeline/dataset.py`:

- определить dataclass или Pydantic-модель `MutationDatasetRow`;
- добавить чтение CSV/TSV через `pandas`;
- валидировать обязательные колонки и canonical amino acids;
- нормализовать `mutation_id`;
- не запускать OpenFold3, FoldX или multiprocessing.
