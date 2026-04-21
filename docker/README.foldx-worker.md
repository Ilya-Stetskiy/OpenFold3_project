# FoldX Docker Worker

## Русский

Этот Docker-образ предназначен только для CPU-запуска FoldX mutation variants. Он не содержит OpenFold GPU worker, CUDA, веса OpenFold, Triton cache или polling-контракт глобального сервера. В image копируется только минимальный Python-код, нужный для FoldX mutation runs.

FoldX не включается в image из-за лицензии. Его нужно скачать отдельно и примонтировать как executable.

### Что нужно

- Docker или Docker Desktop.
- Linux FoldX executable на хосте. Windows `.exe` нельзя запускать внутри Linux container.
- Входная структура `.pdb` или `.cif`.
- Payload JSON с явными мутациями или panel-заданием.

### Сборка

```bash
docker build -f Dockerfile.foldx -t openfold3-foldx-worker:local .
```

### Проверка image без FoldX

```bash
docker run --rm openfold3-foldx-worker:local \
  python3 -m foldx_worker.cli preflight --allow-missing-foldx
```

### Проверка с FoldX

PowerShell:

```powershell
docker run --rm `
  -v D:\foldx-linux\foldx:/foldx/foldx:ro `
  -v D:\foldx_results:/results `
  -v D:\foldx_cache:/cache `
  openfold3-foldx-worker:local `
  python3 -m foldx_worker.cli preflight
```

Linux:

```bash
docker run --rm \
  -v /opt/foldx/foldx:/foldx/foldx:ro \
  -v /data/foldx_results:/results \
  -v /data/foldx_cache:/cache \
  openfold3-foldx-worker:local \
  python3 -m foldx_worker.cli preflight
```

### Запуск явных мутаций

Payload:

```json
{
  "mode": "explicit_mutations",
  "structure_path": "/input/example.pdb",
  "case_id": "example_foldx",
  "mutations": [
    {
      "chain_id": "A",
      "from_residue": "L",
      "position_1based": 42,
      "to_residue": "A"
    }
  ]
}
```

PowerShell:

```powershell
docker run --rm `
  -v D:\foldx-linux\foldx:/foldx/foldx:ro `
  -v D:\foldx_input:/input:ro `
  -v D:\foldx_results:/results `
  -v D:\foldx_cache:/cache `
  openfold3-foldx-worker:local `
  python3 -m foldx_worker.cli run-payload --payload /input/payload.json
```

### Запуск panel по позициям

Payload:

```json
{
  "mode": "panel",
  "structure_path": "/input/example.pdb",
  "chain_id": "A",
  "positions": [42],
  "ranking_metric": "stability_ddg",
  "num_workers": 1
}
```

Panel mode развернет каждую позицию во все 19 не-WT замен и запишет:

- `/results/summary.json`
- `/results/rows.csv`
- `/results/ranking.csv`
- `/results/foldx_worker_manifest.json`
- `/results/cases/...`

### Статусы manifest

- `completed`: все мутации успешно завершились.
- `partial_failed`: часть мутаций завершилась ошибкой; процесс возвращает exit code `1`.
- `failed`: нет успешных мутаций или payload/preflight не прошел; процесс возвращает exit code `1`.

`preflight` теперь не только ищет файл FoldX, но и запускает `foldx --help`, чтобы поймать Windows binary, битый executable или отсутствующие shared libraries до основного расчета.

### Типичные ошибки

- `FoldX binary was not found`: не смонтирован executable или не задан `FOLDX_BINARY`.
- `Permission denied`: FoldX-файл не executable на Linux; выполните `chmod +x`.
- `exec format error`: Windows FoldX binary запущен в Linux container. Нужен Linux FoldX binary.
- `Could not find residue`: chain/residue numbering в payload не совпадает со структурой.
- PowerShell не использует `\` для переноса строк; нужен backtick `` ` ``.

## English

This Docker image is only for CPU FoldX mutation variants. It does not include the OpenFold GPU worker, CUDA, OpenFold weights, Triton cache, or the global server polling contract. The image copies only the minimal Python code required for FoldX mutation runs.

FoldX is not bundled into the image because of licensing. Download it separately and mount the executable into the container.

### Requirements

- Docker or Docker Desktop.
- A Linux FoldX executable on the host. A Windows `.exe` cannot run inside the Linux container.
- An input `.pdb` or `.cif` structure.
- A payload JSON with explicit mutations or a panel job.

### Build

```bash
docker build -f Dockerfile.foldx -t openfold3-foldx-worker:local .
```

### Image check without FoldX

```bash
docker run --rm openfold3-foldx-worker:local \
  python3 -m foldx_worker.cli preflight --allow-missing-foldx
```

### Preflight with FoldX

PowerShell:

```powershell
docker run --rm `
  -v D:\foldx-linux\foldx:/foldx/foldx:ro `
  -v D:\foldx_results:/results `
  -v D:\foldx_cache:/cache `
  openfold3-foldx-worker:local `
  python3 -m foldx_worker.cli preflight
```

Linux:

```bash
docker run --rm \
  -v /opt/foldx/foldx:/foldx/foldx:ro \
  -v /data/foldx_results:/results \
  -v /data/foldx_cache:/cache \
  openfold3-foldx-worker:local \
  python3 -m foldx_worker.cli preflight
```

### Explicit mutations

Use `docker/foldx_payload.example.json` as a template. The worker writes `foldx_worker_manifest.json`, `rows.csv`, and one case directory per mutation.

PowerShell:

```powershell
docker run --rm `
  -v D:\foldx-linux\foldx:/foldx/foldx:ro `
  -v D:\foldx_input:/input:ro `
  -v D:\foldx_results:/results `
  -v D:\foldx_cache:/cache `
  openfold3-foldx-worker:local `
  python3 -m foldx_worker.cli run-payload --payload /input/payload.json
```

### Position panel

Use `docker/foldx_panel_payload.example.json` as a template. Panel mode expands each requested position to all 19 non-WT substitutions and writes `summary.json`, `rows.csv`, and `ranking.csv`.

### Manifest statuses

- `completed`: every mutation finished successfully.
- `partial_failed`: at least one mutation failed; the process exits with code `1`.
- `failed`: no mutation succeeded or payload/preflight failed; the process exits with code `1`.

`preflight` now starts `foldx --help` instead of only checking the file bit. This catches Windows binaries, broken executables, and missing shared libraries before the main run.

### Common errors

- `FoldX binary was not found`: mount the executable or set `FOLDX_BINARY`.
- `Permission denied`: make the Linux FoldX binary executable with `chmod +x`.
- `exec format error`: a Windows FoldX binary was mounted into a Linux container.
- `Could not find residue`: payload chain/residue numbering does not match the structure.
- PowerShell uses the backtick `` ` `` for line continuation, not `\`.
