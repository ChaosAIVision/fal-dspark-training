#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ ! -r "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE; copy .env.example to .env first." >&2
  exit 20
fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

export ROOT
export PROFILE="${PROFILE:-single_gpu}"
export TRAIN_GPU_INDEX="${TRAIN_GPU_INDEX:-1}"
export INFER_GPU_INDEX="${INFER_GPU_INDEX:-0}"
export SHARD_SIZE="${SHARD_SIZE:-128}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export MIN_FREE_GIB="${MIN_FREE_GIB:-28}"
export INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
export INFERENCE_PORT="${INFERENCE_PORT:-30000}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_home}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
