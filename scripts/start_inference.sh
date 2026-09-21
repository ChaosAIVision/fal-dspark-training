#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/load_env.sh"
: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH in .env}"
DRAFT="$ROOT/outputs/$PROFILE/dspark_hf"
[[ -d "$DRAFT" ]] || { echo "Missing exported draft: $DRAFT" >&2; exit 1; }
PYTHON="${SGLANG_PYTHON:-$ROOT/.venv-sglang/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$ROOT/.venv/bin/python"

exec "$ROOT/scripts/gpu_lock.sh" "$INFER_GPU_INDEX" "$PYTHON" -m sglang.launch_server \
  --model-path "$TARGET_MODEL_PATH" --trust-remote-code \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT" \
  --speculative-dspark-block-size 8 --speculative-draft-attention-backend triton \
  --attention-backend triton --tp-size 1 --max-running-requests 1 \
  --context-length "$MAX_SEQ_LENGTH" --mem-fraction-static 0.72 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --host "$INFERENCE_HOST" --port "$INFERENCE_PORT"
