#!/bin/bash
# Launch a single-GPU lingbot-va inference server for LIBERO-Plus eval.
#
# Args:
#   $1  PORT         WebSocket port
#   $2  GPU_ID       physical GPU id (becomes CUDA_VISIBLE_DEVICES)
#   $3  SAVE_ROOT    server-side visualization dir (small files only)
#   $4  CKPT_DIR     pretrained model dir (passed via env, see below)
#
# CKPT_DIR is communicated by setting wan22_pretrained_model_name_or_path
# inside the libero config — but that lives in source. Easiest path: set
# WAN22_CKPT_DIR env var; the wrapper below sed-injects it into a temp copy
# of the libero cfg if it differs from what's already in the file.
#
# Designed to be backgrounded; the caller is responsible for killing it.

set -euo pipefail
PORT="$1"
GPU_ID="$2"
SAVE_ROOT="$3"
CKPT_DIR="${4:-${WAN22_CKPT_DIR:-}}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "ERROR: CKPT_DIR (arg 4 or env WAN22_CKPT_DIR) is required" >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT"

# Patch the libero config to point at the supplied ckpt, idempotently.
# This avoids tracking ckpt-path edits in git.
CFG_FILE="wan_va/configs/va_libero_cfg.py"
[[ -f "$CFG_FILE" ]] || { echo "ERROR: $CFG_FILE not found (run from lingbot-va root)"; exit 1; }
python - "$CFG_FILE" "$CKPT_DIR" <<'PY'
import re, sys
cfg, ckpt = sys.argv[1], sys.argv[2]
src = open(cfg).read()
new = re.sub(
    r'va_libero_cfg\.wan22_pretrained_model_name_or_path\s*=.*',
    f'va_libero_cfg.wan22_pretrained_model_name_or_path = "{ckpt}"',
    src,
)
if new != src:
    open(cfg, 'w').write(new)
    print(f"[launch_server] patched {cfg} → {ckpt}")
else:
    print(f"[launch_server] {cfg} already at {ckpt}")
PY

# Patch attn_mode in the ckpt's transformer/config.json from "flex" → "torch"
# per lingbot-va README warning — required for inference, idempotent.
TRANSFORMER_CFG="$CKPT_DIR/transformer/config.json"
if [[ -f "$TRANSFORMER_CFG" ]]; then
  python - "$TRANSFORMER_CFG" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
if d.get("attn_mode") == "flex":
    d["attn_mode"] = "torch"
    json.dump(d, open(p, "w"), indent=2)
    print(f"[launch_server] patched {p}: attn_mode flex→torch")
else:
    print(f"[launch_server] {p}: attn_mode={d.get('attn_mode')!r} (no patch)")
PY
else
  echo "[launch_server] WARN: $TRANSFORMER_CFG missing — ckpt may not be downloaded yet"
fi

echo "[launch_server] starting server on port=$PORT gpu=$GPU_ID save_root=$SAVE_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$((29500 + PORT % 1000))" \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port "$PORT" \
    --save_root "$SAVE_ROOT"
