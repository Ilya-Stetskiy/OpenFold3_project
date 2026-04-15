#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-openfold3-gpu-worker:local}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.gpu-worker}"
CONTEXT="${CONTEXT:-.}"

docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${CONTEXT}"

docker run --rm "${IMAGE}" \
  python3 -c "import openfold3.run_openfold; import gpu_worker.main; print('import ok')"

docker run --rm "${IMAGE}" \
  python3 -m openfold3.run_openfold --help >/dev/null

docker run --rm "${IMAGE}" \
  run_openfold --help >/dev/null

echo "gpu worker docker smoke ok"
