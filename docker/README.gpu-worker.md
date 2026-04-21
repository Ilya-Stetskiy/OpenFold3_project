# 🇷🇺 README (RU)

# Docker запуск OpenFold3 GPU Worker

## TL;DR

OpenFold3 GPU Worker запускает тяжёлые задачи моделирования на отдельной GPU-машине
и подключается к глобальному веб-серверу только исходящими запросами. Сервер хранит
очередь, а worker берёт одно задание, выполняет OpenFold3, загружает полный архив
результатов и снова становится свободным.

## Overview

Этот Docker-профиль нужен, чтобы безопасно вынести GPU-вычисления из глобального
веб-сервиса. Веб-сервер остаётся центром управления: он принимает пользовательские
запросы, хранит очередь, выдаёт задания worker-ам и принимает результаты.

Worker решает другую задачу: он управляет локальным OpenFold3 runtime на машине с GPU.
Он проверяет доступность окружения, следит за ресурсами, запускает существующий
`run_openfold`, пишет telemetry, сохраняет полный output и передаёт результат обратно
серверу.

Архитектура выбрана как polling, а не входящий worker API. Это проще для GPU-машин за
NAT/firewall, безопаснее для раннего прототипа и не требует открывать вычислительный
узел наружу.

## Features

- Outbound-only worker: GPU-машина сама подключается к серверу.
- Один active GPU job на worker в v1.
- Поддержка одиночного и batch `predict`.
- Поддержка batch вариантов одной базы.
- Point mutation панели используют существующий `screen-mutations`.
- Произвольные варианты или независимые белки используют multi-query `predict`.
- Heartbeat со статусом worker-а и ресурсами.
- Event stream по стадиям job-а.
- Timeout job-а с завершением subprocess tree.
- Полный output OpenFold3 сохраняется и архивируется.
- Локальный TTL cache для отладки и повторной передачи.
- Docker Compose профиль с GPU reservation и volume mounts.

## Key Ideas / Improvements

Главная идея - не переписывать OpenFold3 как веб-сервис. Worker остаётся тонким
оркестратором вокруг уже существующего CLI. Это снижает риск поломать основной
runtime и позволяет сохранить прямой запуск `run_openfold`.

Очередь намеренно не живёт в worker-е. Если очередь будет локальной, потом придётся
синхронизировать пользователей, приоритеты и квоты между сервером и вычислительными
узлами. Поэтому в v1 worker только берёт lease на одно задание и сообщает результат.

Полный output сохраняется, потому что для моделирования важны не только итоговые
метрики. Пользователю и разработчику нужны структуры, confidence files, входные JSON,
логи и telemetry, чтобы сравнивать запуски и разбирать ошибки.

## High-level Pipeline

1. Worker стартует в контейнере.
2. Проверяет директории, OpenFold checkout, свободный диск и конфигурацию.
3. Отправляет heartbeat на сервер.
4. Запрашивает lease.
5. Если задания нет, ждёт следующий polling interval.
6. Если задание есть, создаёт рабочую директорию и пишет `state.json`.
7. Преобразует high-level payload в `query.json` или `screening_job.json`.
8. Запускает `python -m openfold3.run_openfold`.
9. Пишет telemetry во время выполнения.
10. После завершения собирает manifest и архив `.tar.gz`.
11. Загружает архив по upload URL, который пришёл от сервера.
12. Отправляет complete event и возвращается в `idle`.

## Installation

### Требования

На GPU-хосте должны быть установлены:

- Docker с Compose v2.
- NVIDIA driver.
- NVIDIA Container Toolkit.
- Рабочий checkout `OpenFold3_project`.
- Доступные веса OpenFold3.
- Доступные cache-директории для MSA и Triton.
- Python окружение внутри образа или примонтированное conda/venv окружение, из
  которого запускается OpenFold3.

Проверка GPU внутри Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

Если команда не видит GPU, сначала нужно настроить NVIDIA Container Toolkit.

### Обязательные переменные

```bash
export WORKER_TOKEN="secret-token-from-server"
export SERVER_URL="https://your-global-server.example"
```

### Рекомендуемые переменные

```bash
export WORKER_ID="gpu-worker-1"
export PYTHON="python3"
export INSTALL_OPENFOLD_DEPS="1"
export WORKER_CACHE_TTL_DAYS="7"
export MAX_JOB_RUNTIME_SECONDS="86400"
export MIN_FREE_DISK_GB="5"
```

### Volume mounts

Compose-файл по умолчанию монтирует:

```yaml
./:/work/OpenFold3_project
${OPENFOLD_WEIGHTS_DIR:-./.runtime/openfold_weights}:/weights
${OPENFOLD_MSA_CACHE_DIR:-./msa_cache}:/msa_cache
${OPENFOLD_TRITON_CACHE_DIR:-./.runtime/triton_cache}:/triton_cache
${WORKER_RESULTS_HOST_DIR:-./.runtime/gpu_worker}:/work/OpenFold3_project/.runtime/gpu_worker
```

Для реального сервера обычно лучше задать абсолютные host paths:

```bash
export OPENFOLD_WEIGHTS_DIR="/mnt/data/openfold_weights"
export OPENFOLD_MSA_CACHE_DIR="/mnt/data/openfold_msa_cache"
export OPENFOLD_TRITON_CACHE_DIR="/mnt/data/triton_cache"
export WORKER_RESULTS_HOST_DIR="/mnt/data/openfold_worker_results"
```

По умолчанию Dockerfile устанавливает Python-зависимости OpenFold через
`pip install -e ./openfold-3`. Если OpenFold окружение уже собрано на хосте,
можно отключить установку тяжёлых зависимостей при сборке:

```bash
export INSTALL_OPENFOLD_DEPS="0"
```

Затем примонтируйте окружение в контейнер:

```yaml
volumes:
  - /mnt/data/openfold_env:/opt/openfold
```

И укажите:

```bash
export OPENFOLD_PREFIX="/opt/openfold"
export PYTHON="/opt/openfold/bin/python"
```

## Usage

Сборка и запуск из директории `OpenFold3_project`:

```bash
docker compose -f docker-compose.gpu-worker.yml build
docker compose -f docker-compose.gpu-worker.yml up -d
```

Просмотр логов:

```bash
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Остановка:

```bash
docker compose -f docker-compose.gpu-worker.yml down
```

Проверка переменных внутри контейнера:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker env | sort
```

Проверка OpenFold import path:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker \
  python3 -c "import openfold3; print(openfold3.__file__)"
```

Проверка GPU:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker nvidia-smi
```

Проверка локального состояния:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker \
  cat /work/OpenFold3_project/.runtime/gpu_worker/state.json
```

## Project Structure

```text
OpenFold3_project/
  Dockerfile.gpu-worker
  docker-compose.gpu-worker.yml
  .dockerignore
  docker/
    README.gpu-worker.md
  gpu_worker/
    schemas.py
    client.py
    runner.py
    telemetry.py
    artifacts.py
    main.py
```

`Dockerfile.gpu-worker` задаёт CUDA runtime image, устанавливает worker-зависимости
и по умолчанию ставит OpenFold3 в editable-режиме. Для образов, где OpenFold
runtime примонтирован отдельно, используйте `INSTALL_OPENFOLD_DEPS=0`.

`docker-compose.gpu-worker.yml` описывает сервис, GPU reservation, env-переменные и
volume mounts.

`gpu_worker/` содержит код worker-а: контракт, клиент сервера, запуск OpenFold,
телеметрию, упаковку артефактов и главный loop.

## Technical Details

### Архитектура

Worker состоит из четырёх логических слоёв:

- Contract layer: описывает job payloads, limits, events, heartbeat и manifest.
- Server client: выполняет HTTP-запросы к глобальному серверу с Bearer token.
- Execution layer: превращает high-level job в OpenFold input и запускает CLI.
- Artifact layer: собирает output, telemetry, логи, manifest и архив.

Worker не содержит локальную очередь. Он хранит только локальное состояние текущего
или последнего job-а, чтобы после рестарта можно было увидеть, где выполнение
оборвалось.

### Server API

Worker ожидает следующие endpoints:

- `POST /api/workers/heartbeat`
- `POST /api/workers/lease`
- `POST /api/jobs/{job_id}/events`
- `POST /api/jobs/{job_id}/complete`
- upload URL из lease response

Все API-запросы идут с заголовком:

```http
Authorization: Bearer <WORKER_TOKEN>
```

Пустой lease:

```json
{"job": null}
```

Lease с заданием:

```json
{
  "job": {
    "job_id": "job-123",
    "lease_id": "lease-456",
    "job_type": "predict_batch",
    "payload": {},
    "limits": {
      "max_runtime_seconds": 3600,
      "min_free_disk_gb": 5,
      "max_queries": 8,
      "max_variants": 32
    },
    "upload": {
      "url": "https://your-global-server.example/uploads/job-123",
      "method": "PUT",
      "headers": {}
    }
  }
}
```

### `predict_batch`

`predict_batch` предназначен для одного или нескольких независимых queries.

Минимальный payload:

```json
{
  "queries": [
    {
      "query_id": "ubiquitin",
      "molecules": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "ACDE"
        }
      ]
    }
  ],
  "num_diffusion_samples": 1,
  "num_model_seeds": 1,
  "use_msa_server": false,
  "use_templates": false
}
```

Worker генерирует `query.json` и запускает:

```bash
python -m openfold3.run_openfold predict \
  --query_json <work_dir>/query.json \
  --output_dir <work_dir>/output
```

### `variant_batch`

`variant_batch` предназначен для одной базы и набора вариантов.

Если каждый вариант содержит ровно одну point mutation, worker использует
`screen-mutations`, чтобы сохранить существующую логику кэша и batch orchestration.

Пример:

```json
{
  "base_query": {
    "molecules": [
      {
        "molecule_type": "protein",
        "chain_ids": ["A"],
        "sequence": "ACDE"
      }
    ]
  },
  "query_prefix": "scan",
  "variants": [
    {
      "variant_id": "A_C2G",
      "mutations": [
        {
          "chain_id": "A",
          "position_1based": 2,
          "to_residue": "G"
        }
      ]
    }
  ]
}
```

Worker генерирует `screening_job.json` и запускает:

```bash
python -m openfold3.run_openfold screen-mutations \
  --screening_job_json <work_dir>/screening_job.json
```

Если варианты заданы полными молекулами или содержат несколько изменений, worker
собирает multi-query `query.json` и использует обычный `predict`.

### Локальные файлы результата

В `WORKER_RESULTS_DIR` создаётся:

```text
state.json
<job_id>/
  query.json
  screening_job.json
  run_openfold.log
  screen_mutations.log
  telemetry.csv
  artifact_manifest.json
  output/
  screening/
<job_id>.tar.gz
```

Фактический набор файлов зависит от типа job-а. Архив включает рабочую директорию
целиком.

### Runtime limits

Worker применяет:

- `max_runtime_seconds` из lease или `MAX_JOB_RUNTIME_SECONDS`.
- `min_free_disk_gb` из lease или `MIN_FREE_DISK_GB`.
- `max_queries` для `predict_batch`.
- `max_variants` для `variant_batch`.

При timeout worker завершает subprocess group и помечает job как failed.

### Performance considerations

- В v1 worker выполняет один GPU job, чтобы не конфликтовать за VRAM.
- Point mutation панели используют `screen-mutations`, потому что там уже есть
  переиспользование кэша и batch preparation.
- Полный output может быстро занимать много места, поэтому нужен отдельный results
  volume и разумный `WORKER_CACHE_TTL_DAYS`.
- Telemetry interval не должен быть слишком маленьким в production, чтобы не создавать
  лишний IO.

## Limitations

- Worker выполняет только один active GPU job.
- Локальной очереди нет.
- Pause/cancel не реализованы.
- Worker не предоставляет входящий HTTP API.
- Тяжёлые данные, веса и cache-директории всё ещё должны приходить через volume
  mounts.

## Future Work

- Добавить cancel через серверный control endpoint.
- Добавить upload retry с backoff и resume.
- Добавить поддержку нескольких GPU как `1 active job per GPU`.
- Добавить server-side signed URLs для объектного хранилища.
- Добавить отдельный smoke-test compose profile с fake server.
- Добавить healthcheck для контейнера.

---

# 🇬🇧 README (EN)

# Docker Setup for OpenFold3 GPU Worker

## TL;DR

OpenFold3 GPU Worker runs heavy structure-prediction jobs on a dedicated GPU
machine and talks to the global web server only through outbound requests. The
server owns the queue; the worker leases one job, runs OpenFold3, uploads a full
result archive, and returns to idle.

## Overview

This Docker profile exists to move GPU computation out of the global web server.
The web server remains the control plane: it accepts user requests, stores the
queue, leases work to workers, and receives completed artifacts.

The worker manages the local OpenFold3 runtime on the GPU node. It validates the
environment, tracks resources, runs the existing `run_openfold` CLI, records
telemetry, preserves the full output, and sends the result back to the server.

The design uses polling instead of an inbound worker API. That is safer for early
deployment, works better behind NAT/firewalls, and avoids exposing the GPU machine
to the public network.

## Features

- Outbound-only worker.
- One active GPU job per worker in v1.
- Single-query and multi-query `predict` support.
- Variant batches based on one input.
- Point mutation panels reuse the existing `screen-mutations` path.
- Arbitrary variants or independent proteins use multi-query `predict`.
- Heartbeat with worker status and resource metrics.
- Job lifecycle events.
- Job timeout with subprocess-tree termination.
- Full OpenFold3 output preservation.
- Local TTL cache for debugging and re-upload.
- Docker Compose profile with GPU reservation and volume mounts.

## Key Ideas / Improvements

The main idea is to avoid turning OpenFold3 itself into a web server. The worker is
a thin orchestration layer around the existing CLI. This reduces risk and keeps
direct `run_openfold` usage intact.

The queue intentionally does not live in the worker. A local queue would later
require synchronization of users, priorities, quotas, and fairness across the
server and all GPU nodes. In v1 the worker only leases one job and reports the
result.

The full output is preserved because structure-prediction jobs are not fully
represented by one metric. Users and developers need structures, confidence files,
input JSON, logs, and telemetry to compare runs and debug failures.

## High-level Pipeline

1. The worker starts in a container.
2. It validates directories, the OpenFold checkout, disk space, and configuration.
3. It sends a heartbeat to the server.
4. It requests a lease.
5. If no job is available, it waits for the next polling interval.
6. If a job is available, it creates a work directory and writes `state.json`.
7. It converts the high-level payload into `query.json` or `screening_job.json`.
8. It runs `python -m openfold3.run_openfold`.
9. It records telemetry while the job runs.
10. It writes a manifest and builds a `.tar.gz` archive.
11. It uploads the archive to the upload URL provided by the server.
12. It sends completion status and returns to `idle`.

## Installation

### Requirements

The GPU host needs:

- Docker with Compose v2.
- NVIDIA driver.
- NVIDIA Container Toolkit.
- A working `OpenFold3_project` checkout.
- Available OpenFold3 model weights.
- Available MSA and Triton cache directories.
- A Python environment inside the image, or a mounted conda/venv environment that
  can run OpenFold3.

Check GPU access from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

If this command cannot see the GPU, configure NVIDIA Container Toolkit first.

### Required variables

```bash
export WORKER_TOKEN="secret-token-from-server"
export SERVER_URL="https://your-global-server.example"
```

### Recommended variables

```bash
export WORKER_ID="gpu-worker-1"
export PYTHON="python3"
export INSTALL_OPENFOLD_DEPS="1"
export WORKER_CACHE_TTL_DAYS="7"
export MAX_JOB_RUNTIME_SECONDS="86400"
export MIN_FREE_DISK_GB="5"
```

### Volume mounts

The default Compose file mounts:

```yaml
./:/work/OpenFold3_project
${OPENFOLD_WEIGHTS_DIR:-./.runtime/openfold_weights}:/weights
${OPENFOLD_MSA_CACHE_DIR:-./msa_cache}:/msa_cache
${OPENFOLD_TRITON_CACHE_DIR:-./.runtime/triton_cache}:/triton_cache
${WORKER_RESULTS_HOST_DIR:-./.runtime/gpu_worker}:/work/OpenFold3_project/.runtime/gpu_worker
```

For a real server, absolute host paths are usually better:

```bash
export OPENFOLD_WEIGHTS_DIR="/mnt/data/openfold_weights"
export OPENFOLD_MSA_CACHE_DIR="/mnt/data/openfold_msa_cache"
export OPENFOLD_TRITON_CACHE_DIR="/mnt/data/triton_cache"
export WORKER_RESULTS_HOST_DIR="/mnt/data/openfold_worker_results"
```

By default, the Dockerfile installs OpenFold Python dependencies with
`pip install -e ./openfold-3`. If an OpenFold environment already exists on the
host, you can skip the heavy dependency install during image build:

```bash
export INSTALL_OPENFOLD_DEPS="0"
```

Then mount that environment into the container:

```yaml
volumes:
  - /mnt/data/openfold_env:/opt/openfold
```

Then set:

```bash
export OPENFOLD_PREFIX="/opt/openfold"
export PYTHON="/opt/openfold/bin/python"
```

## Usage

Build and start from `OpenFold3_project`:

```bash
docker compose -f docker-compose.gpu-worker.yml build
docker compose -f docker-compose.gpu-worker.yml up -d
```

Follow logs:

```bash
docker compose -f docker-compose.gpu-worker.yml logs -f gpu-worker
```

Stop:

```bash
docker compose -f docker-compose.gpu-worker.yml down
```

Inspect container environment:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker env | sort
```

Check the OpenFold import path:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker \
  python3 -c "import openfold3; print(openfold3.__file__)"
```

Check GPU visibility:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker nvidia-smi
```

Inspect local worker state:

```bash
docker compose -f docker-compose.gpu-worker.yml exec gpu-worker \
  cat /work/OpenFold3_project/.runtime/gpu_worker/state.json
```

## Project Structure

```text
OpenFold3_project/
  Dockerfile.gpu-worker
  docker-compose.gpu-worker.yml
  .dockerignore
  docker/
    README.gpu-worker.md
  gpu_worker/
    schemas.py
    client.py
    runner.py
    telemetry.py
    artifacts.py
    main.py
```

`Dockerfile.gpu-worker` defines the CUDA runtime image, installs worker
dependencies, and installs OpenFold3 in editable mode by default. For images
where the OpenFold runtime is mounted separately, use `INSTALL_OPENFOLD_DEPS=0`.

`docker-compose.gpu-worker.yml` defines the service, GPU reservation, environment
variables, and volume mounts.

`gpu_worker/` contains the worker code: contract models, server client, OpenFold
execution, telemetry, artifact packaging, and the main loop.

## Technical Details

### Architecture

The worker has four logical layers:

- Contract layer: job payloads, limits, events, heartbeat, and manifest.
- Server client: HTTP calls to the global server with a Bearer token.
- Execution layer: high-level job conversion and OpenFold CLI execution.
- Artifact layer: output, telemetry, logs, manifest, and archive packaging.

The worker does not maintain a local queue. It only stores local state for the
current or most recent job so that interrupted runs can be inspected after restart.

### Server API

The worker expects these endpoints:

- `POST /api/workers/heartbeat`
- `POST /api/workers/lease`
- `POST /api/jobs/{job_id}/events`
- `POST /api/jobs/{job_id}/complete`
- upload URL from the lease response

All API requests include:

```http
Authorization: Bearer <WORKER_TOKEN>
```

Empty lease:

```json
{"job": null}
```

Lease with a job:

```json
{
  "job": {
    "job_id": "job-123",
    "lease_id": "lease-456",
    "job_type": "predict_batch",
    "payload": {},
    "limits": {
      "max_runtime_seconds": 3600,
      "min_free_disk_gb": 5,
      "max_queries": 8,
      "max_variants": 32
    },
    "upload": {
      "url": "https://your-global-server.example/uploads/job-123",
      "method": "PUT",
      "headers": {}
    }
  }
}
```

### `predict_batch`

`predict_batch` is used for one or more independent queries.

Minimal payload:

```json
{
  "queries": [
    {
      "query_id": "ubiquitin",
      "molecules": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "ACDE"
        }
      ]
    }
  ],
  "num_diffusion_samples": 1,
  "num_model_seeds": 1,
  "use_msa_server": false,
  "use_templates": false
}
```

The worker writes `query.json` and runs:

```bash
python -m openfold3.run_openfold predict \
  --query_json <work_dir>/query.json \
  --output_dir <work_dir>/output
```

### `variant_batch`

`variant_batch` is used for one base query and a set of variants.

If each variant contains exactly one point mutation, the worker uses
`screen-mutations` to preserve the existing cache and batch orchestration logic.

Example:

```json
{
  "base_query": {
    "molecules": [
      {
        "molecule_type": "protein",
        "chain_ids": ["A"],
        "sequence": "ACDE"
      }
    ]
  },
  "query_prefix": "scan",
  "variants": [
    {
      "variant_id": "A_C2G",
      "mutations": [
        {
          "chain_id": "A",
          "position_1based": 2,
          "to_residue": "G"
        }
      ]
    }
  ]
}
```

The worker writes `screening_job.json` and runs:

```bash
python -m openfold3.run_openfold screen-mutations \
  --screening_job_json <work_dir>/screening_job.json
```

If variants are full molecule definitions or contain multiple changes, the worker
builds a multi-query `query.json` and uses regular `predict`.

### Local output files

`WORKER_RESULTS_DIR` contains:

```text
state.json
<job_id>/
  query.json
  screening_job.json
  run_openfold.log
  screen_mutations.log
  telemetry.csv
  artifact_manifest.json
  output/
  screening/
<job_id>.tar.gz
```

The exact files depend on the job type. The archive includes the whole work
directory.

### Runtime limits

The worker enforces:

- `max_runtime_seconds` from the lease or `MAX_JOB_RUNTIME_SECONDS`.
- `min_free_disk_gb` from the lease or `MIN_FREE_DISK_GB`.
- `max_queries` for `predict_batch`.
- `max_variants` for `variant_batch`.

On timeout, the worker terminates the subprocess group and marks the job as failed.

### Performance considerations

- In v1 the worker runs one GPU job to avoid VRAM contention.
- Point mutation panels use `screen-mutations` because it already has cache reuse and
  batch preparation.
- Full output can consume disk quickly, so use a dedicated results volume and a
  reasonable `WORKER_CACHE_TTL_DAYS`.
- Production telemetry interval should not be too low to avoid unnecessary IO.

## Limitations

- Only one active GPU job per worker.
- No local queue.
- No pause/cancel support yet.
- No inbound worker HTTP API.
- Large data, model weights, and cache directories still need to be provided
  through volume mounts.

## Future Work

- Add cancel support through a server-side control endpoint.
- Add upload retry with backoff and resume.
- Support multiple GPUs as one active job per GPU.
- Support server-side signed URLs for object storage.
- Add a smoke-test Compose profile with a fake server.
- Add a container healthcheck.
