#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: gpu_lock.sh <physical-index> <command> [args...]" >&2
  exit 64
fi

gpu_index="$1"
shift
expected_var="EXPECTED_TRAIN_GPU_UUID"
[[ "$gpu_index" == "${INFER_GPU_INDEX:-}" ]] && expected_var="EXPECTED_INFER_GPU_UUID"
expected="${!expected_var:-}"
actual="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

if [[ -n "$expected" && "$actual" != "$expected" ]]; then
  echo "Refusing to run: GPU $gpu_index UUID is $actual, expected $expected" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_index"
echo "Locked to physical GPU $gpu_index ($actual); visible as logical cuda:0." >&2
exec "$@"
