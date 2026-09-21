#!/usr/bin/env bash
set -euo pipefail
session="${INFERENCE_TMUX_SESSION:-dspark-infer}"
if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
  echo "Stopped tmux session $session"
else
  echo "No tmux session named $session"
fi
