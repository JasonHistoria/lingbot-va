#!/bin/bash
#SBATCH --job-name=lvva_lp_libero10
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=256G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=24:00:00
#SBATCH --output=./logs/lvva_lp_libero10_%j.out
#SBATCH --error=./logs/lvva_lp_libero10_%j.err

# Sub-LIBERO-Plus eval (libero_10 perturbation subset, 420 tasks = 7 cats × 60)
# of the lingbot-va-posttrain-libero-long release ckpt. 2× H200, 1 trial/task.
# Sub task list mirrors FastWAM's generate_subeval_tasks.py output filtered
# to libero_10 — same 60 evenly-spaced task_ids per category as FastWAM, so
# results are directly comparable.
#
# Mirrors FastWAM/scripts/eval_sub_libero_plus_joint_34d_rotvec_h200.slurm
# but runs lingbot-va's WebSocket server/client architecture instead of
# FastWAM's in-process eval.
#
# Usage:
#   sbatch scripts/eval_libero_plus_libero10_h200.slurm
#
# All paths are wired against tillicum + the lingbot-va env produced by
# scripts/setup_lingbot_va_tillicum.sh. Override any of the env vars below
# in the sbatch invocation if your layout differs:
#   sbatch --export=ALL,RUN_DIR=/some/where scripts/eval_libero_plus_libero10_h200.slurm

set -euo pipefail

# ── Paths (override via --export if needed) ──────────────────────────────────
LINGBOT_VA_ROOT="${LINGBOT_VA_ROOT:-/gpfs/home/jliu63/projects/lingbot-va}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/gpfs/home/jliu63/projects/HVLA/world_model/FastWAM}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/gpfs/home/jliu63/projects/HVLA/third_party/LIBERO-plus}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-/gpfs/home/jliu63/activate_lingbot_va_tillicum.sh}"
RUN_DIR="${RUN_DIR:-/gpfs/home/jliu63/projects/lingbot-va/runs}"

[[ -f "$ACTIVATE_SCRIPT" ]] || { echo "ERROR: $ACTIVATE_SCRIPT missing — run scripts/setup_lingbot_va_tillicum.sh first"; exit 1; }
source "$ACTIVATE_SCRIPT"
cd "$LINGBOT_VA_ROOT"
mkdir -p logs

# ── Download release ckpt (idempotent) ───────────────────────────────────────
CKPT_DIR="$RUN_DIR/lingbot-va-posttrain-libero-long"
if [[ ! -f "$CKPT_DIR/transformer/config.json" ]]; then
  echo "[slurm] downloading robbyant/lingbot-va-posttrain-libero-long → $CKPT_DIR"
  mkdir -p "$RUN_DIR"
  huggingface-cli download robbyant/lingbot-va-posttrain-libero-long \
    --local-dir "$CKPT_DIR"
else
  echo "[slurm] ckpt already present at $CKPT_DIR"
fi

# ── LIBERO-Plus environment (NOT vanilla LIBERO) ─────────────────────────────
# Put LIBERO-plus first on PYTHONPATH so `from libero.libero import benchmark`
# resolves to the perturbation suite. Server+client both inherit this PYTHONPATH.
export PYTHONPATH="${LIBERO_PLUS_ROOT}:.:${PYTHONPATH:-}"
rm -rf ~/.libero
# First-time benchmark import answers a Y/N stdin prompt with "N" so client
# subshells don't get hung on it. Non-zero rc here is fine.
python -c "from libero.libero import benchmark" <<< "N" 2>&1 | tail -3 || true

# ── Generate the sub libero_10 task file (idempotent) ────────────────────────
# 7 categories × 60 evenly-spaced task_ids each = 420 tasks. Same 60-per-cell
# selection as FastWAM's generate_subeval_tasks.py, filtered to libero_10.
TASK_FILE="${LINGBOT_VA_ROOT}/logs/tasks_libero_plus_sub_libero10.txt"
if [[ ! -f "$TASK_FILE" ]]; then
  echo "[slurm] generating sub task list → $TASK_FILE"
  TMP_FULL=$(mktemp)
  python "${FASTWAM_ROOT}/experiments/libero/generate_subeval_tasks.py" \
    --classification_path "${LIBERO_PLUS_ROOT}/libero/libero/benchmark/task_classification.json" \
    --output_plus "$TMP_FULL" \
    --n_per_cell 60
  grep "^libero_10," "$TMP_FULL" > "$TASK_FILE"
  rm -f "$TMP_FULL"
fi
echo "[slurm] TASK_FILE: $TASK_FILE  ($(wc -l < $TASK_FILE) tasks)"

# ── Output directory: eval_results/<run_tag>_<timestamp>/ ────────────────────
RUN_TAG="lingbot_va_libero_long_sub_libero_plus_libero10"
OUTPUT_DIR="${LINGBOT_VA_ROOT}/eval_results/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Sub-LIBERO-Plus libero_10 eval (lingbot-va-posttrain-libero-long)"
echo "  CKPT_DIR:   $CKPT_DIR"
echo "  TASK_FILE:  $TASK_FILE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  Started:    $(date)"
echo "========================================================"

# ── Run sharded eval (2 GPUs, paired server+client per GPU) ──────────────────
export NUM_GPUS=2
export CKPT_DIR
export TASK_FILE
export OUTPUT_DIR
export PROMPT_SOURCE=libero_plus_smart
export SAVE_VIDEOS=0
export BASE_PORT=29056
export SERVER_WARMUP_SEC=120  # 14B params + VAE + T5 takes a while

bash evaluation/libero_plus/run_libero_plus_batch.sh

# ── Summarize across 7 perturbation dimensions × libero_10 ───────────────────
echo ""
echo "========================================================"
echo "Summarizing results..."
python "${FASTWAM_ROOT}/experiments/libero/summarize_results_plus.py" \
  --output_dir "$OUTPUT_DIR" \
  --classification_json "${LIBERO_PLUS_ROOT}/libero/libero/benchmark/task_classification.json"

echo "Eval done: $(date)"
