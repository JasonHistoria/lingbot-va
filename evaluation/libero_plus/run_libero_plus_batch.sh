#!/bin/bash
# Sharded multi-GPU LIBERO-Plus eval for lingbot-va.
#
# Splits TASK_FILE round-robin across NUM_GPUS shards, then for each GPU
# starts ONE inference server + ONE eval client (paired by port). Servers
# are killed once all clients finish.
#
# Required env:
#   NUM_GPUS, CKPT_DIR, TASK_FILE, OUTPUT_DIR
# Optional env:
#   BASE_PORT (default 29056)        first port; subsequent GPUs use BASE_PORT+i
#   PROMPT_SOURCE (default libero_plus_smart)
#   MAX_TIMESTEPS (default 800)
#   SAVE_VIDEOS (default 0)
#   SERVER_WARMUP_SEC (default 60)   how long to wait after launching servers
#                                    before starting clients

set -euo pipefail

: "${NUM_GPUS:?need NUM_GPUS}"
: "${CKPT_DIR:?need CKPT_DIR}"
: "${TASK_FILE:?need TASK_FILE}"
: "${OUTPUT_DIR:?need OUTPUT_DIR}"

BASE_PORT="${BASE_PORT:-29056}"
PROMPT_SOURCE="${PROMPT_SOURCE:-libero_plus_smart}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-800}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-60}"

mkdir -p "$OUTPUT_DIR/shards" "$OUTPUT_DIR/server_logs" "$OUTPUT_DIR/client_logs"

# ─── 1. Round-robin shard the task file ──────────────────────────────────────
python3 - "$TASK_FILE" "$NUM_GPUS" "$OUTPUT_DIR/shards" <<'PY'
import os, sys
task_file, num_gpus, out_root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = [l for l in (x.strip() for x in open(task_file)) if l and not l.startswith("#")]
print(f"[shard] {len(lines)} tasks → {num_gpus} GPUs")
shards = [[] for _ in range(num_gpus)]
for i, line in enumerate(lines):
    shards[i % num_gpus].append(line)
for g, shard in enumerate(shards):
    p = os.path.join(out_root, f"shard_gpu{g}.txt")
    with open(p, "w") as f:
        f.write("\n".join(shard) + "\n")
    print(f"  GPU {g}: {len(shard)} tasks → {p}")
PY

# ─── 2. Launch one server per GPU ────────────────────────────────────────────
SERVER_PIDS=()
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  PORT="$((BASE_PORT + GPU))"
  SAVE_ROOT="$OUTPUT_DIR/server_logs/gpu${GPU}_visualization"
  LOG="$OUTPUT_DIR/server_logs/gpu${GPU}.log"
  echo "[batch] starting server GPU=$GPU PORT=$PORT → $LOG"
  bash evaluation/libero_plus/launch_server.sh "$PORT" "$GPU" "$SAVE_ROOT" "$CKPT_DIR" \
       > "$LOG" 2>&1 &
  SERVER_PIDS+=($!)
done

# Best-effort cleanup if we get killed
cleanup() {
  echo "[batch] cleanup: killing servers ${SERVER_PIDS[*]}"
  for pid in "${SERVER_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${SERVER_PIDS[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "[batch] waiting ${SERVER_WARMUP_SEC}s for servers to load model..."
sleep "$SERVER_WARMUP_SEC"

# Sanity-check servers are alive before launching clients
for i in "${!SERVER_PIDS[@]}"; do
  pid="${SERVER_PIDS[$i]}"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[batch] FATAL: server GPU=$i (pid $pid) died during warmup. See server log."
    exit 1
  fi
done

# ─── 3. Launch one client per GPU, wait for all ──────────────────────────────
CLIENT_PIDS=()
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  PORT="$((BASE_PORT + GPU))"
  SHARD="$OUTPUT_DIR/shards/shard_gpu${GPU}.txt"
  LOG="$OUTPUT_DIR/client_logs/gpu${GPU}.log"
  echo "[batch] starting client GPU=$GPU PORT=$PORT shard=$SHARD → $LOG"
  PROMPT_SOURCE="$PROMPT_SOURCE" MAX_TIMESTEPS="$MAX_TIMESTEPS" SAVE_VIDEOS="$SAVE_VIDEOS" \
    nohup bash evaluation/libero_plus/launch_client.sh "$PORT" "$SHARD" "$OUTPUT_DIR" \
         > "$LOG" 2>&1 &
  CLIENT_PIDS+=($!)
done

echo "[batch] launched ${#CLIENT_PIDS[@]} clients. Tail with:  tail -f $OUTPUT_DIR/client_logs/gpu0.log"

FAIL=0
for i in "${!CLIENT_PIDS[@]}"; do
  if wait "${CLIENT_PIDS[$i]}"; then
    echo "[batch] client GPU=$i OK"
  else
    rc=$?
    echo "[batch] client GPU=$i FAILED rc=$rc (see $OUTPUT_DIR/client_logs/gpu${i}.log)"
    FAIL=$((FAIL + 1))
  fi
done

if [ "$FAIL" -gt 0 ]; then
  echo "[batch] $FAIL client(s) failed"
  exit 1
fi
echo "[batch] all clients OK"
