#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/load_env.sh"
: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH in .env}"

SHARDS="${SHARDS_DIR:-$ROOT/data/shards}"
OFFLINE="$ROOT/data/offline_work"
STATE="$ROOT/state/$PROFILE"
CURRENT="$ROOT/outputs/$PROFILE/dspark/checkpoints"
PREVIOUS="$ROOT/outputs/$PROFILE/dspark/checkpoints.previous"
WORK="$ROOT/outputs/$PROFILE/dspark_work"
EXPORT="$ROOT/outputs/$PROFILE/dspark_hf"
LOG="$ROOT/logs/sharded-training.log"
mkdir -p "$STATE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
valid_checkpoint() {
  local root="$1" iteration
  [[ -s "$root/latest_checkpointed_iteration.txt" ]] || return 1
  iteration="$(<"$root/latest_checkpointed_iteration.txt")"
  [[ "$iteration" =~ ^[0-9]+$ ]] || return 1
  [[ -f "$root/iter_$(printf '%07d' "$iteration")/model/.metadata" ]]
}
safe_remove() {
  local path="${1%/}"
  case "$path" in "$OFFLINE"|"$CURRENT"|"$PREVIOUS"|"$WORK"|"$EXPORT"|"$EXPORT.previous") rm -rf -- "$path";;
    *) log "Unsafe cleanup target: $path"; exit 90;; esac
}
free_gib() { df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}'; }
guard_disk() { (( $(free_gib) >= "$1" )) || { log "Disk guard failed: $(free_gib) GiB free"; exit 30; }; }
gpu() { "$ROOT/scripts/gpu_lock.sh" "$TRAIN_GPU_INDEX" "$@"; }

[[ -s "$SHARDS/manifest.json" ]] || { log "Prepare dataset shards first: $SHARDS"; exit 21; }
valid_checkpoint "$CURRENT" || { log "Missing valid initial checkpoint: $CURRENT"; exit 22; }
mapfile -t files < <("$ROOT/.venv/bin/python" -c \
  'import json,sys; print(*[x["file"] for x in json.load(open(sys.argv[1]))["shards"]], sep="\n")' \
  "$SHARDS/manifest.json")

log "Starting/resuming ${#files[@]} shards on physical GPU $TRAIN_GPU_INDEX"
for name in "${files[@]}"; do
  id="${name%.jsonl}"
  [[ -f "$STATE/$id.done" ]] && { log "SKIP $id"; continue; }
  if [[ ! -d "$CURRENT" && -d "$PREVIOUS" ]]; then
    if [[ -f "$STATE/$id.trained" ]] && valid_checkpoint "$WORK/checkpoints"; then
      mv "$WORK/checkpoints" "$CURRENT"
      safe_remove "$PREVIOUS"
      touch "$STATE/$id.done"
      rm -f "$STATE/$id.trained"
      log "RECOVERED $id after checkpoint promotion interruption"
      continue
    fi
    mv "$PREVIOUS" "$CURRENT"
  fi
  guard_disk "$MIN_FREE_GIB"
  safe_remove "$OFFLINE"
  safe_remove "$WORK"
  log "MATERIALIZE $id"
  gpu "$ROOT/.venv/bin/python" "$ROOT/pipeline.py" materialize \
    --target "$TARGET_MODEL_PATH" --data "$SHARDS/$name" --offline-data "$OFFLINE" \
    --max-seq-length "$MAX_SEQ_LENGTH" --overwrite
  log "TRAIN $id"
  gpu "$ROOT/.venv/bin/python" "$ROOT/pipeline.py" train \
    --target "$TARGET_MODEL_PATH" --data "$SHARDS/$name" --offline-data "$OFFLINE" \
    --load-path "$CURRENT" --output-dir "$WORK" --max-seq-length "$MAX_SEQ_LENGTH"
  valid_checkpoint "$WORK/checkpoints" || { log "Invalid new checkpoint for $id"; exit 23; }
  touch "$STATE/$id.trained"
  safe_remove "$PREVIOUS"
  mv "$CURRENT" "$PREVIOUS"
  mv "$WORK/checkpoints" "$CURRENT"
  if ! valid_checkpoint "$CURRENT"; then
    rm -rf -- "$CURRENT"
    mv "$PREVIOUS" "$CURRENT"
    exit 24
  fi
  safe_remove "$PREVIOUS"
  touch "$STATE/$id.done"
  rm -f "$STATE/$id.trained"
  safe_remove "$OFFLINE"
  safe_remove "$WORK"
  log "DONE $id; disk=$(free_gib)GiB"
done

log "EXPORT"
safe_remove "$EXPORT.previous"
if [[ -d "$EXPORT" ]]; then mv "$EXPORT" "$EXPORT.previous"; fi
if ! gpu "$ROOT/.venv/bin/python" "$ROOT/pipeline.py" export \
  --target "$TARGET_MODEL_PATH" --load-path "$CURRENT" --output-dir "$EXPORT"; then
  safe_remove "$EXPORT"
  [[ -d "$EXPORT.previous" ]] && mv "$EXPORT.previous" "$EXPORT"
  exit 25
fi
safe_remove "$EXPORT.previous"
touch "$STATE/pipeline.done"
log "PIPELINE COMPLETE"
