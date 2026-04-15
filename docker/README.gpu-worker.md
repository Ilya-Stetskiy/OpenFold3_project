# 🇷🇺 README (RU)

# OpenFold3 GPU Worker Docker Guide

## TL;DR

Этот Docker-образ запускает OpenFold3 GPU worker как самостоятельный runtime: исходный код и Python-зависимости находятся внутри image, а с хоста монтируются только веса, cache и результаты. Worker может работать в двух режимах: локальный smoke test без сервера и production polling mode с глобальным сервером.

Проверенный минимальный путь: собрать image, положить checkpoint в `/weights`, пройти `--preflight-only`, затем запустить `gpu_worker.local_run` с тестовым payload. Успешный smoke test должен закончиться `local job finished with status=completed`.

## Overview

GPU worker нужен для выноса тяжелого OpenFold3 inference из веб-сервера на отдельную GPU-машину. В production сервер хранит очередь и выдает lease, а worker сам подключается к серверу исходящими HTTP-запросами, берет одно задание, запускает OpenFold3, архивирует полный output и отправляет результат назад.

Docker-сборка сделана как self-contained runtime. Compose больше не монтирует локальный checkout поверх `/work/OpenFold3_project`, поэтому чистая машина не зависит от незакоммиченных файлов рядом с Dockerfile. Это важно для воспроизводимости: image содержит код worker-а, OpenFold3 package, CUDA/PyTorch runtime, CUTLASS и системные зависимости.

Локальный режим нужен до подключения реального сервера. Он запускает один payload через тот же worker pipeline, пишет events, telemetry, manifest и имитирует upload копированием архива в `/results/local_uploads`.

## Features

- Self-contained Docker image на базе NVIDIA PyTorch.
- OpenFold3 устанавливается внутри контейнера.
- Source bind mount не используется по умолчанию.
- Веса монтируются в `/weights` и реально используются через `OPENFOLD_CACHE=/weights`.
- Автоматический поиск checkpoint: `of3-p2-155k.pt`, `of3-p2-145k.pt`, `of3_ft3_v1.pt`.
- `OPENFOLD_INFERENCE_CKPT_PATH` позволяет явно указать checkpoint.
- Preflight проверяет OpenFold paths, torch import, CUDA, `nvidia-smi`, checkpoint и `run_openfold --help`.
- Standalone local-run без `SERVER_URL` и `WORKER_TOKEN`.
- Production polling mode с `WORKER_TOKEN` и `SERVER_URL`.
- Полный output, generated input JSON, logs, telemetry, manifest и archive сохраняются в results.
- Timeout и failure cases тоже архивируются.
- Worker держит не больше одного активного OpenFold job.

## Key Ideas / Improvements

Главная идея - не превращать OpenFold3 в отдельный web API внутри GPU-машины. Worker остается тонкой orchestration-обвязкой вокруг существующего CLI `run_openfold`. Это сохраняет прямую совместимость с основными командами OpenFold3 и снижает риск расхождения поведения между CLI и веб-сервисом.

Вторая идея - отделить immutable runtime от mutable data. Image содержит код и зависимости, а host volumes содержат только веса, MSA cache, Triton cache и результаты. Поэтому одна и та же сборка может запускаться на чистой машине без локального checkout.

Третья идея - сначала проверять локально. `gpu_worker.local_run` позволяет поймать проблемы с CUDA, checkpoint, RAM, bind mounts и OpenFold imports до подключения server polling contract.

## High-level Pipeline

1. Docker image собирается из repository checkout.
2. В container image копируются `openfold-3` и `gpu_worker`.
3. В runtime монтируются `/weights`, `/results`, `/triton_cache`, `/msa_cache`.
4. Preflight проверяет Python/OpenFold/CUDA/checkpoint/CLI.
5. В local mode worker читает payload JSON и создает synthetic lease.
6. В server mode worker отправляет heartbeat и запрашивает lease у сервера.
7. Worker пишет `query.json` или `screening_job.json`.
8. Worker запускает `python3 -m openfold3.run_openfold predict` или `screen-mutations`.
9. Во время выполнения пишутся telemetry и job events.
10. После завершения создается `artifact_manifest.json`.
11. Рабочая директория архивируется в `.tar.gz`.
12. Local mode копирует архив в `/results/local_uploads`; server mode загружает его по `upload_url`.
13. Worker отправляет complete/failure metadata и возвращается в idle.

## Installation

### 1. Требования

На Linux GPU-сервере:

- NVIDIA driver.
- Docker Engine с Compose v2.
- NVIDIA Container Toolkit.
- Доступ к GitHub repository.
- Доступ к OpenFold checkpoint.
- Достаточно RAM и disk space.

На Windows laptop/workstation:

- Docker Desktop.
- WSL 2 backend.
- NVIDIA driver с WSL GPU support.
- PowerShell.
- Достаточно RAM, выделенной WSL через `.wslconfig`.

Если на cloud-сервере нет `docker`, `podman`, `apptainer`, `singularity` или `nerdctl`, build на этой машине не пройдет без установки container runtime администратором. Наличие GPU само по себе не заменяет Docker/Container Toolkit.

### 2. Клонирование

Если SSH key не настроен и появляется `Permission denied (publickey)`, используйте HTTPS:

```powershell
git clone --branch openfold_docker https://github.com/Ilya-Stetskiy/OpenFold3_project.git OpenFold3_project_docker_test
cd OpenFold3_project_docker_test
```

Linux:

```bash
git clone --branch openfold_docker https://github.com/Ilya-Stetskiy/OpenFold3_project.git OpenFold3_project_docker_test
cd OpenFold3_project_docker_test
```

### 3. Папки для runtime data

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force D:\openfold_weights
New-Item -ItemType Directory -Force D:\openfold_results
New-Item -ItemType Directory -Force D:\openfold_triton_cache
New-Item -ItemType Directory -Force D:\openfold_msa_cache
```

Linux:

```bash
mkdir -p /data/openfold_weights /data/openfold_results /data/openfold_triton_cache /data/openfold_msa_cache
```

Пустые папки достаточны для results/cache, но `/weights` должен содержать checkpoint перед настоящим predict.

### 4. WSL memory на Windows

Если Docker Desktop показывает, что resource limits управляются Windows/WSL, настройка памяти делается не в UI Docker Desktop, а через `%USERPROFILE%\.wslconfig`.

Проверить RAM:

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object @{Name="TotalGB";Expression={"{0:N1}" -f ($_.TotalVisibleMemorySize/1MB)}},
                @{Name="FreeGB";Expression={"{0:N1}" -f ($_.FreePhysicalMemory/1MB)}}
```

Для ноутбука с 16 GB RAM практичный минимум:

```powershell
@"
[wsl2]
memory=12GB
processors=8
swap=16GB
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding ASCII
```

Применить:

```powershell
wsl --shutdown
```

Затем перезапустите Docker Desktop и проверьте:

```powershell
docker run --rm openfold3-gpu-worker:local bash -lc "free -h"
```

Если до этого был `OpenFold exited with code -9`, почти всегда причина в нехватке RAM или слишком маленьком WSL memory limit.

### 5. Сборка image

Windows PowerShell и Linux одинаково:

```powershell
docker build -f Dockerfile.gpu-worker -t openfold3-gpu-worker:local .
```

Если нужно полностью исключить старый build cache:

```powershell
docker build --no-cache -f Dockerfile.gpu-worker -t openfold3-gpu-worker:local .
```

После небольших правок исходников обычно используйте build без `--no-cache`: Docker переиспользует тяжелые слои base image, apt, pip и CUTLASS.

### 6. Checkpoint

Worker ищет checkpoint в `/weights` через `OPENFOLD_CACHE=/weights`. Поддерживаемые имена по умолчанию:

- `of3-p2-155k.pt`
- `of3-p2-145k.pt`
- `of3_ft3_v1.pt`

Если checkpoint уже есть на host, положите его в папку weights.

Windows:

```powershell
Get-ChildItem D:\openfold_weights
```

Linux:

```bash
ls -lah /data/openfold_weights
```

Если checkpoint нужно скачать через OpenFold setup script, на Windows PowerShell можно передать ответы интерактивному скрипту:

```powershell
"/weights`n/weights`n1`nno" | docker run --rm -i `
  -v D:\openfold_weights:/weights `
  -e OPENFOLD_CACHE=/weights `
  openfold3-gpu-worker:local `
  python3 -m openfold3.setup_openfold download --skip_confirmation
```

Linux:

```bash
printf "/weights\n/weights\n1\nno\n" | docker run --rm -i \
  -v /data/openfold_weights:/weights \
  -e OPENFOLD_CACHE=/weights \
  openfold3-gpu-worker:local \
  python3 -m openfold3.setup_openfold download --skip_confirmation
```

Ожидаемый результат - файл вроде `/weights/of3-p2-155k.pt` и файл `/weights/ckpt_root`.

## Usage

### 1. Базовые smoke checks

Проверить import:

```powershell
docker run --rm openfold3-gpu-worker:local python3 -c "import openfold3.run_openfold; import gpu_worker.main; print('import ok')"
```

Проверить CLI:

```powershell
docker run --rm openfold3-gpu-worker:local run_openfold --help
```

Проверить GPU:

```powershell
docker run --rm --gpus all openfold3-gpu-worker:local nvidia-smi
```

Предупреждение `CUDA Minor Version Compatibility mode ENABLED` не обязательно фатальное. Оно означает, что driver и CUDA runtime не идеально совпадают. Для production лучше обновить NVIDIA driver, но если preflight и predict проходят, тестовый запуск валиден.

### 2. Preflight

Windows PowerShell:

```powershell
docker run --rm --gpus all `
  -v D:\openfold_weights:/weights `
  -v D:\openfold_results:/results `
  -v D:\openfold_triton_cache:/triton_cache `
  -v D:\openfold_msa_cache:/msa_cache `
  openfold3-gpu-worker:local `
  python3 -m gpu_worker.main --preflight-only
```

Linux:

```bash
docker run --rm --gpus all \
  -v /data/openfold_weights:/weights \
  -v /data/openfold_results:/results \
  -v /data/openfold_triton_cache:/triton_cache \
  -v /data/openfold_msa_cache:/msa_cache \
  openfold3-gpu-worker:local \
  python3 -m gpu_worker.main --preflight-only
```

Успешный результат:

```text
gpu_worker preflight ok
```

`WORKER_TOKEN` для `--preflight-only` не нужен.

### 3. Local worker run без сервера

Windows PowerShell:

```powershell
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
  -v D:\openfold_weights:/weights `
  -v D:\openfold_results:/results `
  -v D:\openfold_triton_cache:/triton_cache `
  -v D:\openfold_msa_cache:/msa_cache `
  -v ${PWD}\docker\local_payload.example.json:/payload.json:ro `
  openfold3-gpu-worker:local `
  python3 -m gpu_worker.local_run --payload /payload.json --results-dir /results
```

Linux:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/openfold_weights:/weights \
  -v /data/openfold_results:/results \
  -v /data/openfold_triton_cache:/triton_cache \
  -v /data/openfold_msa_cache:/msa_cache \
  -v "$PWD/docker/local_payload.example.json:/payload.json:ro" \
  openfold3-gpu-worker:local \
  python3 -m gpu_worker.local_run --payload /payload.json --results-dir /results
```

Успешный результат:

```text
local job finished with status=completed: /results/local-...
```

Проверить manifest и log:

```powershell
$job = Get-ChildItem D:\openfold_results -Directory |
  Where-Object { $_.Name -like "local-*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content D:\openfold_results\local_completed_manifest.json
Get-Content "$($job.FullName)\run_openfold.log" -Tail 80
```

Критерии успеха:

- manifest содержит `"status": "completed"`;
- manifest содержит `"error": null`;
- log содержит `GPU available: True (cuda), used: True`;
- log содержит `Successful Queries: 1` и `Failed Queries: 0`;
- в `D:\openfold_results\local_uploads` или `/data/openfold_results/local_uploads` есть свежий `.tar.gz`.

### 4. Production server mode

Windows PowerShell:

```powershell
$env:WORKER_ID="gpu-worker-1"
$env:WORKER_TOKEN="secret-token-from-server"
$env:SERVER_URL="https://your-server.example"
$env:OPENFOLD_WEIGHTS_DIR="D:\openfold_weights"
$env:OPENFOLD_MSA_CACHE_DIR="D:\openfold_msa_cache"
$env:OPENFOLD_TRITON_CACHE_DIR="D:\openfold_triton_cache"
$env:WORKER_RESULTS_HOST_DIR="D:\openfold_results"

docker compose -f docker-compose.gpu-worker.yml up -d --build gpu-worker
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Linux:

```bash
export WORKER_ID="gpu-worker-1"
export WORKER_TOKEN="secret-token-from-server"
export SERVER_URL="https://your-server.example"
export OPENFOLD_WEIGHTS_DIR="/data/openfold_weights"
export OPENFOLD_MSA_CACHE_DIR="/data/openfold_msa_cache"
export OPENFOLD_TRITON_CACHE_DIR="/data/openfold_triton_cache"
export WORKER_RESULTS_HOST_DIR="/data/openfold_results"

docker compose -f docker-compose.gpu-worker.yml up -d --build gpu-worker
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Остановить:

```bash
docker compose -f docker-compose.gpu-worker.yml down
```

## Project Structure

```text
OpenFold3_project/
  Dockerfile.gpu-worker
  docker-compose.gpu-worker.yml
  docker/
    README.gpu-worker.md
    local_payload.example.json
    smoke_gpu_worker.sh
  gpu_worker/
    schemas.py
    client.py
    runner.py
    telemetry.py
    artifacts.py
    local_run.py
    main.py
  openfold-3/
    openfold3/
      run_openfold.py
      core/data/tools/
```

`Dockerfile.gpu-worker` собирает self-contained CUDA/PyTorch/OpenFold runtime.

`docker-compose.gpu-worker.yml` описывает production service и local profile без source bind mount.

`gpu_worker/local_run.py` запускает один payload без внешнего сервера и завершает процесс с ненулевым кодом, если manifest failed.

`gpu_worker/main.py` содержит daemon loop, preflight и server polling mode.

`gpu_worker/runner.py` преобразует payload в OpenFold CLI invocation.

`openfold-3/openfold3/core/data/tools/` должен быть в Git. Если эта папка отсутствует, `run_openfold predict` падает с `ModuleNotFoundError: No module named 'openfold3.core.data.tools'`.

## Technical Details

### Runtime paths

Внутри контейнера используются стабильные пути:

```text
/work/OpenFold3_project      source code inside image
/work/OpenFold3_project/openfold-3
/weights                    checkpoints and OpenFold cache
/results                    worker outputs and manifests
/triton_cache               Triton and torch extension cache
/msa_cache                  MSA cache
```

Основные env-переменные:

```text
OPENFOLD_CACHE=/weights
OPENFOLD_INFERENCE_CKPT_PATH=
WORKER_RESULTS_DIR=/results
TRITON_CACHE_DIR=/triton_cache
TORCH_EXTENSIONS_DIR=/triton_cache/torch_extensions
WORKER_REQUIRE_CUDA=1
WORKER_REQUIRE_CHECKPOINT=1
MAX_JOB_RUNTIME_SECONDS=86400
MIN_FREE_DISK_GB=5
```

Если нужно указать checkpoint явно:

```powershell
-e OPENFOLD_INFERENCE_CKPT_PATH=/weights/of3-p2-155k.pt
```

### Worker-server contract

Production mode использует:

- `POST /api/workers/heartbeat`
- `POST /api/workers/lease`
- `POST /api/jobs/{job_id}/events`
- upload URL из lease response
- `POST /api/jobs/{job_id}/complete`

Все server calls используют:

```http
Authorization: Bearer <WORKER_TOKEN>
```

### Job types

`predict_batch`:

- high-level molecules/queries;
- worker пишет `query.json`;
- worker запускает `run_openfold predict`.

`variant_batch`:

- base query плюс variants;
- single point-mutation panels идут через `screen-mutations`;
- arbitrary variants идут через multi-query `predict`.

### Troubleshooting

`docker: command not found`:

- Docker не установлен или не в PATH.
- На managed cloud без прав администратора Docker может быть недоступен.
- Нужен Docker/Podman/Apptainer/Singularity/nerdctl или prebuilt image на машине с container runtime.

`git@github.com: Permission denied (publickey)`:

- SSH key не настроен.
- Используйте HTTPS clone.

`/opt/nvidia/nvidia_entrypoint.sh: exec: \: not found`:

- В PowerShell использован Linux line continuation `\`.
- В PowerShell используйте backtick: `` ` ``.

`Python was not found` после Docker-команды:

- Часть команды выполнилась на host Windows, а не внутри container.
- Обычно причина - неправильный перенос строки в PowerShell.

`WORKER_TOKEN must be set`:

- Это нормально для production mode без токена.
- Для локального smoke используйте `python3 -m gpu_worker.local_run`.
- Для preflight используйте `python3 -m gpu_worker.main --preflight-only`.

`FileNotFoundError: /usr/local/cuda/bin/nvcc`:

- Старый image был собран на CUDA runtime base без `nvcc`.
- Новый Dockerfile использует NVIDIA PyTorch base. Пересоберите image.

`No OpenFold checkpoint found`:

- `/weights` пустой или mounted не туда.
- Проверьте host path и наличие `of3-p2-155k.pt`.
- Можно задать `OPENFOLD_INFERENCE_CKPT_PATH`.

`ModuleNotFoundError: No module named 'openfold3.core.data.tools'`:

- В image не попала папка `openfold-3/openfold3/core/data/tools`.
- Проверьте, что она не игнорируется Git и присутствует в checkout перед build.

`OpenFold exited with code -9`:

- Обычно процесс был убит из-за RAM pressure.
- На Windows/WSL настройте `.wslconfig`.
- Увеличьте memory/swap, закройте тяжелые приложения, повторите run.

`WARNING: SHMEM allocation limit is 64MB`:

- Для реальных запусков добавляйте:

```text
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```

`CUDA Minor Version Compatibility mode ENABLED`:

- Driver и CUDA runtime не идеально совпадают.
- Если тест проходит, warning не блокирует smoke.
- Для production лучше обновить NVIDIA driver.

`DataLoader will create 10 worker processes`:

- Warning не обязательно фатальный.
- На слабых машинах может замедлять запуск; это отдельная настройка OpenFold runtime.

## Limitations

- GPU worker v1 не содержит локальную очередь.
- Один worker process управляет одним GPU execution slot.
- Cancel/pause не реализованы; timeout убивает subprocess tree.
- Local mode не заменяет server integration test, он проверяет только standalone pipeline.
- На 4 GB VRAM и 16 GB RAM возможны ограничения по размеру реальных задач.
- Windows Docker Desktop через WSL требует отдельной настройки memory/swap.
- `setup_openfold download` остается интерактивным, поэтому в Docker-примерах ввод передается через pipe.
- Full output archive может быть большим на реальных задачах.

## Future Work

- Добавить официальный fake polling server для end-to-end server contract smoke.
- Добавить retry/backoff для upload failures.
- Добавить chunked/streaming upload для больших архивов.
- Добавить более подробную классификацию OpenFold exit codes.
- Добавить настройки числа DataLoader workers для малых машин.
- Подготовить prebuilt image publishing pipeline.
- Добавить отдельный minimal test payload для variant/screen-mutations.
- Добавить server-side dashboard для worker telemetry и job progress.

---

# 🇬🇧 README (EN)

# OpenFold3 GPU Worker Docker Guide

## TL;DR

This Docker image runs the OpenFold3 GPU worker as a self-contained runtime: source code and Python dependencies live inside the image, while the host only provides weights, caches, and results. The worker supports two modes: a local smoke test without a server and production polling mode against the global server.

The verified minimum path is: build the image, place a checkpoint under `/weights`, run `--preflight-only`, then run `gpu_worker.local_run` with the test payload. A successful smoke test ends with `local job finished with status=completed`.

## Overview

The GPU worker moves heavy OpenFold3 inference out of the web server and onto a dedicated GPU machine. In production, the global server owns the queue and leases jobs; the worker connects outbound, takes one lease, runs OpenFold3, packages the full output, and sends the result back.

The Docker setup is designed as a self-contained runtime. Compose no longer bind-mounts the local checkout over `/work/OpenFold3_project`, so a clean machine does not depend on uncommitted files next to the Dockerfile. The image contains the worker code, OpenFold3 package, CUDA/PyTorch runtime, CUTLASS, and system dependencies.

The local mode exists to validate the machine before connecting it to a real server. It runs one payload through the same worker pipeline, writes events, telemetry, a manifest, and simulates upload by copying the archive into `/results/local_uploads`.

## Features

- Self-contained Docker image based on NVIDIA PyTorch.
- OpenFold3 is installed inside the container.
- No default source bind mount.
- Weights are mounted at `/weights` and used through `OPENFOLD_CACHE=/weights`.
- Automatic checkpoint lookup for `of3-p2-155k.pt`, `of3-p2-145k.pt`, and `of3_ft3_v1.pt`.
- `OPENFOLD_INFERENCE_CKPT_PATH` can point to an explicit checkpoint.
- Preflight checks OpenFold paths, torch import, CUDA, `nvidia-smi`, checkpoint, and `run_openfold --help`.
- Standalone local-run without `SERVER_URL` or `WORKER_TOKEN`.
- Production polling mode with `WORKER_TOKEN` and `SERVER_URL`.
- Full output, generated input JSON, logs, telemetry, manifest, and archive are retained.
- Timeout and failure cases are archived as well.
- The worker runs at most one active OpenFold job.

## Key Ideas / Improvements

The main idea is not to turn OpenFold3 into a separate web API on the GPU machine. The worker stays a thin orchestration layer around the existing `run_openfold` CLI. This preserves direct CLI compatibility and reduces the risk of behavior drift between command-line usage and the web-service workflow.

The second idea is to separate immutable runtime from mutable data. The image contains code and dependencies; host volumes contain only weights, MSA cache, Triton cache, and results. The same image can therefore run on a clean machine without a local source checkout.

The third idea is to test locally first. `gpu_worker.local_run` catches CUDA, checkpoint, RAM, bind mount, and OpenFold import issues before the server polling contract is introduced.

## High-level Pipeline

1. The Docker image is built from the repository checkout.
2. `openfold-3` and `gpu_worker` are copied into the image.
3. Runtime mounts provide `/weights`, `/results`, `/triton_cache`, and `/msa_cache`.
4. Preflight validates Python, OpenFold, CUDA, checkpoint, and CLI availability.
5. In local mode, the worker reads a payload JSON and creates a synthetic lease.
6. In server mode, the worker sends heartbeat and requests a lease.
7. The worker writes `query.json` or `screening_job.json`.
8. The worker runs `python3 -m openfold3.run_openfold predict` or `screen-mutations`.
9. Telemetry and job events are written during execution.
10. `artifact_manifest.json` is created after completion or failure.
11. The work directory is packaged into a `.tar.gz` archive.
12. Local mode copies the archive into `/results/local_uploads`; server mode uploads it to the provided `upload_url`.
13. The worker sends completion or failure metadata and returns to idle.

## Installation

### 1. Requirements

On a Linux GPU server:

- NVIDIA driver.
- Docker Engine with Compose v2.
- NVIDIA Container Toolkit.
- Access to the GitHub repository.
- Access to an OpenFold checkpoint.
- Enough RAM and disk space.

On a Windows laptop or workstation:

- Docker Desktop.
- WSL 2 backend.
- NVIDIA driver with WSL GPU support.
- PowerShell.
- Enough RAM allocated to WSL through `.wslconfig`.

If a cloud server has no `docker`, `podman`, `apptainer`, `singularity`, or `nerdctl`, the build cannot run there unless an administrator installs a container runtime. Having a GPU does not replace Docker or the NVIDIA Container Toolkit.

### 2. Clone

If SSH keys are not configured and you see `Permission denied (publickey)`, use HTTPS:

```powershell
git clone --branch openfold_docker https://github.com/Ilya-Stetskiy/OpenFold3_project.git OpenFold3_project_docker_test
cd OpenFold3_project_docker_test
```

Linux:

```bash
git clone --branch openfold_docker https://github.com/Ilya-Stetskiy/OpenFold3_project.git OpenFold3_project_docker_test
cd OpenFold3_project_docker_test
```

### 3. Runtime data directories

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force D:\openfold_weights
New-Item -ItemType Directory -Force D:\openfold_results
New-Item -ItemType Directory -Force D:\openfold_triton_cache
New-Item -ItemType Directory -Force D:\openfold_msa_cache
```

Linux:

```bash
mkdir -p /data/openfold_weights /data/openfold_results /data/openfold_triton_cache /data/openfold_msa_cache
```

Empty directories are enough for results and caches, but `/weights` must contain a checkpoint before a real prediction.

### 4. WSL memory on Windows

If Docker Desktop says resource limits are managed by Windows/WSL, memory is not configured in the Docker Desktop UI. Configure `%USERPROFILE%\.wslconfig` instead.

Check system RAM:

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object @{Name="TotalGB";Expression={"{0:N1}" -f ($_.TotalVisibleMemorySize/1MB)}},
                @{Name="FreeGB";Expression={"{0:N1}" -f ($_.FreePhysicalMemory/1MB)}}
```

For a 16 GB laptop, a practical minimum is:

```powershell
@"
[wsl2]
memory=12GB
processors=8
swap=16GB
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding ASCII
```

Apply the settings:

```powershell
wsl --shutdown
```

Restart Docker Desktop and verify:

```powershell
docker run --rm openfold3-gpu-worker:local bash -lc "free -h"
```

If you previously saw `OpenFold exited with code -9`, the usual cause is RAM pressure or a WSL memory limit that is too low.

### 5. Build the image

Windows PowerShell and Linux:

```powershell
docker build -f Dockerfile.gpu-worker -t openfold3-gpu-worker:local .
```

To ignore old build cache completely:

```powershell
docker build --no-cache -f Dockerfile.gpu-worker -t openfold3-gpu-worker:local .
```

After small source changes, usually build without `--no-cache`: Docker will reuse the heavy base image, apt, pip, and CUTLASS layers.

### 6. Checkpoint

The worker looks for checkpoints in `/weights` through `OPENFOLD_CACHE=/weights`. Default supported names:

- `of3-p2-155k.pt`
- `of3-p2-145k.pt`
- `of3_ft3_v1.pt`

If a checkpoint already exists on the host, place it in the weights directory.

Windows:

```powershell
Get-ChildItem D:\openfold_weights
```

Linux:

```bash
ls -lah /data/openfold_weights
```

If you need to download the checkpoint through the OpenFold setup script, PowerShell can pipe answers into the interactive script:

```powershell
"/weights`n/weights`n1`nno" | docker run --rm -i `
  -v D:\openfold_weights:/weights `
  -e OPENFOLD_CACHE=/weights `
  openfold3-gpu-worker:local `
  python3 -m openfold3.setup_openfold download --skip_confirmation
```

Linux:

```bash
printf "/weights\n/weights\n1\nno\n" | docker run --rm -i \
  -v /data/openfold_weights:/weights \
  -e OPENFOLD_CACHE=/weights \
  openfold3-gpu-worker:local \
  python3 -m openfold3.setup_openfold download --skip_confirmation
```

The expected result is a file such as `/weights/of3-p2-155k.pt` plus `/weights/ckpt_root`.

## Usage

### 1. Basic smoke checks

Check imports:

```powershell
docker run --rm openfold3-gpu-worker:local python3 -c "import openfold3.run_openfold; import gpu_worker.main; print('import ok')"
```

Check CLI:

```powershell
docker run --rm openfold3-gpu-worker:local run_openfold --help
```

Check GPU visibility:

```powershell
docker run --rm --gpus all openfold3-gpu-worker:local nvidia-smi
```

`CUDA Minor Version Compatibility mode ENABLED` is not always fatal. It means the host driver and container CUDA runtime do not perfectly match. For production, update the NVIDIA driver if possible; for smoke testing, a passing preflight and prediction are sufficient.

### 2. Preflight

Windows PowerShell:

```powershell
docker run --rm --gpus all `
  -v D:\openfold_weights:/weights `
  -v D:\openfold_results:/results `
  -v D:\openfold_triton_cache:/triton_cache `
  -v D:\openfold_msa_cache:/msa_cache `
  openfold3-gpu-worker:local `
  python3 -m gpu_worker.main --preflight-only
```

Linux:

```bash
docker run --rm --gpus all \
  -v /data/openfold_weights:/weights \
  -v /data/openfold_results:/results \
  -v /data/openfold_triton_cache:/triton_cache \
  -v /data/openfold_msa_cache:/msa_cache \
  openfold3-gpu-worker:local \
  python3 -m gpu_worker.main --preflight-only
```

Successful output:

```text
gpu_worker preflight ok
```

`WORKER_TOKEN` is not required for `--preflight-only`.

### 3. Local worker run without a server

Windows PowerShell:

```powershell
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
  -v D:\openfold_weights:/weights `
  -v D:\openfold_results:/results `
  -v D:\openfold_triton_cache:/triton_cache `
  -v D:\openfold_msa_cache:/msa_cache `
  -v ${PWD}\docker\local_payload.example.json:/payload.json:ro `
  openfold3-gpu-worker:local `
  python3 -m gpu_worker.local_run --payload /payload.json --results-dir /results
```

Linux:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/openfold_weights:/weights \
  -v /data/openfold_results:/results \
  -v /data/openfold_triton_cache:/triton_cache \
  -v /data/openfold_msa_cache:/msa_cache \
  -v "$PWD/docker/local_payload.example.json:/payload.json:ro" \
  openfold3-gpu-worker:local \
  python3 -m gpu_worker.local_run --payload /payload.json --results-dir /results
```

Successful output:

```text
local job finished with status=completed: /results/local-...
```

Check manifest and logs:

```powershell
$job = Get-ChildItem D:\openfold_results -Directory |
  Where-Object { $_.Name -like "local-*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content D:\openfold_results\local_completed_manifest.json
Get-Content "$($job.FullName)\run_openfold.log" -Tail 80
```

Success criteria:

- the manifest contains `"status": "completed"`;
- the manifest contains `"error": null`;
- the log contains `GPU available: True (cuda), used: True`;
- the log contains `Successful Queries: 1` and `Failed Queries: 0`;
- a fresh `.tar.gz` exists in `D:\openfold_results\local_uploads` or `/data/openfold_results/local_uploads`.

### 4. Production server mode

Windows PowerShell:

```powershell
$env:WORKER_ID="gpu-worker-1"
$env:WORKER_TOKEN="secret-token-from-server"
$env:SERVER_URL="https://your-server.example"
$env:OPENFOLD_WEIGHTS_DIR="D:\openfold_weights"
$env:OPENFOLD_MSA_CACHE_DIR="D:\openfold_msa_cache"
$env:OPENFOLD_TRITON_CACHE_DIR="D:\openfold_triton_cache"
$env:WORKER_RESULTS_HOST_DIR="D:\openfold_results"

docker compose -f docker-compose.gpu-worker.yml up -d --build gpu-worker
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Linux:

```bash
export WORKER_ID="gpu-worker-1"
export WORKER_TOKEN="secret-token-from-server"
export SERVER_URL="https://your-server.example"
export OPENFOLD_WEIGHTS_DIR="/data/openfold_weights"
export OPENFOLD_MSA_CACHE_DIR="/data/openfold_msa_cache"
export OPENFOLD_TRITON_CACHE_DIR="/data/openfold_triton_cache"
export WORKER_RESULTS_HOST_DIR="/data/openfold_results"

docker compose -f docker-compose.gpu-worker.yml up -d --build gpu-worker
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Stop:

```bash
docker compose -f docker-compose.gpu-worker.yml down
```

## Project Structure

```text
OpenFold3_project/
  Dockerfile.gpu-worker
  docker-compose.gpu-worker.yml
  docker/
    README.gpu-worker.md
    local_payload.example.json
    smoke_gpu_worker.sh
  gpu_worker/
    schemas.py
    client.py
    runner.py
    telemetry.py
    artifacts.py
    local_run.py
    main.py
  openfold-3/
    openfold3/
      run_openfold.py
      core/data/tools/
```

`Dockerfile.gpu-worker` builds the self-contained CUDA, PyTorch, and OpenFold runtime.

`docker-compose.gpu-worker.yml` defines the production service and local profile without a source bind mount.

`gpu_worker/local_run.py` runs one payload without an external server and exits with a non-zero status if the manifest failed.

`gpu_worker/main.py` contains the daemon loop, preflight, and server polling mode.

`gpu_worker/runner.py` converts payloads into OpenFold CLI invocations.

`openfold-3/openfold3/core/data/tools/` must be tracked in Git. If this package is missing, `run_openfold predict` fails with `ModuleNotFoundError: No module named 'openfold3.core.data.tools'`.

## Technical Details

### Runtime paths

Stable paths inside the container:

```text
/work/OpenFold3_project      source code inside image
/work/OpenFold3_project/openfold-3
/weights                    checkpoints and OpenFold cache
/results                    worker outputs and manifests
/triton_cache               Triton and torch extension cache
/msa_cache                  MSA cache
```

Key environment variables:

```text
OPENFOLD_CACHE=/weights
OPENFOLD_INFERENCE_CKPT_PATH=
WORKER_RESULTS_DIR=/results
TRITON_CACHE_DIR=/triton_cache
TORCH_EXTENSIONS_DIR=/triton_cache/torch_extensions
WORKER_REQUIRE_CUDA=1
WORKER_REQUIRE_CHECKPOINT=1
MAX_JOB_RUNTIME_SECONDS=86400
MIN_FREE_DISK_GB=5
```

To force an explicit checkpoint:

```powershell
-e OPENFOLD_INFERENCE_CKPT_PATH=/weights/of3-p2-155k.pt
```

### Worker-server contract

Production mode uses:

- `POST /api/workers/heartbeat`
- `POST /api/workers/lease`
- `POST /api/jobs/{job_id}/events`
- upload URL from the lease response
- `POST /api/jobs/{job_id}/complete`

All server calls use:

```http
Authorization: Bearer <WORKER_TOKEN>
```

### Job types

`predict_batch`:

- high-level molecules and queries;
- the worker writes `query.json`;
- the worker runs `run_openfold predict`.

`variant_batch`:

- one base query plus variants;
- single point-mutation panels use `screen-mutations`;
- arbitrary variants use multi-query `predict`.

### Troubleshooting

`docker: command not found`:

- Docker is not installed or not in PATH.
- On a managed cloud server without administrator access, Docker may be unavailable.
- You need Docker, Podman, Apptainer, Singularity, nerdctl, or a prebuilt image on a machine with a container runtime.

`git@github.com: Permission denied (publickey)`:

- SSH keys are not configured.
- Use HTTPS clone.

`/opt/nvidia/nvidia_entrypoint.sh: exec: \: not found`:

- Linux line continuation `\` was used in PowerShell.
- In PowerShell, use a backtick: `` ` ``.

`Python was not found` after a Docker command:

- Part of the command ran on the Windows host instead of inside the container.
- The usual cause is incorrect line continuation in PowerShell.

`WORKER_TOKEN must be set`:

- This is expected for production mode without a token.
- For local smoke, use `python3 -m gpu_worker.local_run`.
- For preflight, use `python3 -m gpu_worker.main --preflight-only`.

`FileNotFoundError: /usr/local/cuda/bin/nvcc`:

- An old image was built on a CUDA runtime base without `nvcc`.
- The current Dockerfile uses the NVIDIA PyTorch base. Rebuild the image.

`No OpenFold checkpoint found`:

- `/weights` is empty or mounted incorrectly.
- Check the host path and make sure `of3-p2-155k.pt` exists.
- You can set `OPENFOLD_INFERENCE_CKPT_PATH`.

`ModuleNotFoundError: No module named 'openfold3.core.data.tools'`:

- The image is missing `openfold-3/openfold3/core/data/tools`.
- Make sure the package is not ignored by Git and exists in the checkout before building.

`OpenFold exited with code -9`:

- The process was usually killed because of RAM pressure.
- On Windows/WSL, configure `.wslconfig`.
- Increase memory and swap, close heavy applications, and rerun.

`WARNING: SHMEM allocation limit is 64MB`:

- For real runs, add:

```text
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864
```

`CUDA Minor Version Compatibility mode ENABLED`:

- The host driver and container CUDA runtime do not perfectly match.
- If the test passes, this warning does not block the smoke test.
- For production, update the NVIDIA driver if possible.

`DataLoader will create 10 worker processes`:

- This warning is not necessarily fatal.
- On smaller machines it may slow execution; tuning DataLoader workers is a separate OpenFold runtime setting.

## Limitations

- GPU worker v1 has no local queue.
- One worker process controls one GPU execution slot.
- Cancel and pause are not implemented; timeout kills the subprocess tree.
- Local mode is not a replacement for server integration testing; it only validates the standalone pipeline.
- 4 GB VRAM and 16 GB RAM limit the size of practical real-world jobs.
- Docker Desktop on Windows through WSL requires separate memory and swap configuration.
- `setup_openfold download` remains interactive, so Docker examples pipe answers into it.
- Full output archives can be large for real jobs.

## Future Work

- Add an official fake polling server for end-to-end server contract smoke tests.
- Add retry and backoff for upload failures.
- Add chunked or streaming upload for large archives.
- Add more precise OpenFold exit-code classification.
- Add DataLoader worker tuning for small machines.
- Prepare a prebuilt image publishing pipeline.
- Add a minimal variant/screen-mutations test payload.
- Add a server-side dashboard for worker telemetry and job progress.
