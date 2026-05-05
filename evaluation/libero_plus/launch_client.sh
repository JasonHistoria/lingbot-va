#!/bin/bash
# Launch a single LIBERO-Plus eval client against an already-running server.
#
# Args:
#   $1  PORT         WebSocket port (same as server)
#   $2  TASKS_FILE   tasks file with `suite,task_id` per line
#   $3  OUTPUT_DIR   per-task results JSON dir
# Env (optional):
#   PROMPT_SOURCE       default libero_plus_smart
#   MAX_TIMESTEPS       default 800
#   SAVE_VIDEOS         set to 1 to save per-task rollout MP4

set -euo pipefail
PORT="$1"
TASKS_FILE="$2"
OUTPUT_DIR="$3"

PROMPT_SOURCE="${PROMPT_SOURCE:-libero_plus_smart}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-800}"

ARGS=(
  --tasks-file "$TASKS_FILE"
  --port "$PORT"
  --output-dir "$OUTPUT_DIR"
  --prompt-source "$PROMPT_SOURCE"
  --max-timesteps "$MAX_TIMESTEPS"
)
if [[ "${SAVE_VIDEOS:-0}" == "1" ]]; then
  ARGS+=(--save-videos)
fi

python evaluation/libero_plus/client.py "${ARGS[@]}"
