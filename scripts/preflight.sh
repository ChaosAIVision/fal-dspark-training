#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/load_env.sh"

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH in .env}"
command -v nvidia-smi >/dev/null
command -v tmux >/dev/null
[[ -x "$ROOT/.venv/bin/python" ]] || { echo "Run ./scripts/bootstrap.sh first" >&2; exit 1; }
[[ -d "$ROOT/upstream/TorchSpec" ]] || { echo "TorchSpec checkout missing" >&2; exit 1; }

available="$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
(( available >= MIN_FREE_GIB )) || { echo "Only ${available} GiB free; need ${MIN_FREE_GIB}" >&2; exit 1; }

"$ROOT/scripts/gpu_lock.sh" "$TRAIN_GPU_INDEX" "$ROOT/.venv/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

echo "Preflight passed: ${available} GiB free; training GPU $TRAIN_GPU_INDEX is available."
