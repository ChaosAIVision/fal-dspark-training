#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/scripts/load_env.sh"
total=0
manifest="$ROOT/data/shards/manifest.json"
[[ -s "$manifest" ]] && total="$($ROOT/.venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"]))' "$manifest" 2>/dev/null || echo 0)"
done_count="$(find "$ROOT/state/$PROFILE" -name 'shard-*.done' -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "shards=$done_count/$total"
tmux list-sessions 2>/dev/null | grep -E 'dspark-(train|infer)' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
df -h "$ROOT" | tail -1
curl -fsS "http://$INFERENCE_HOST:$INFERENCE_PORT/health" >/dev/null 2>&1 && echo "inference=healthy" || echo "inference=offline"
