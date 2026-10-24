#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/run_profiles

if tmux has-session -t openfold3_cpu_heavy 2>/dev/null; then
  echo "CPU tmux session already exists."
else
  tmux new-session -d -s openfold3_cpu_heavy "python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run --no-progress-bar 2>&1 | tee logs/run_profiles/cpu_heavy_latest.log"
  echo "Started openfold3_cpu_heavy"
fi

if tmux has-session -t openfold3_gpu_heavy 2>/dev/null; then
  echo "GPU tmux session already exists."
else
  tmux new-session -d -s openfold3_gpu_heavy "python3 scripts/run/run_stage_group.py --stage-group gpu_heavy --dataset current_internal --dry-run --no-progress-bar 2>&1 | tee logs/run_profiles/gpu_heavy_latest.log"
  echo "Started openfold3_gpu_heavy"
fi

echo "tmux attach -t openfold3_cpu_heavy"
echo "tmux attach -t openfold3_gpu_heavy"
echo "tail -f logs/run_profiles/cpu_heavy_latest.log"
echo "tail -f logs/run_profiles/gpu_heavy_latest.log"
