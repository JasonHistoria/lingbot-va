# LIBERO-Plus eval for lingbot-va

Sharded WebSocket-server/client eval that runs `robbyant/lingbot-va-posttrain-libero-long` on the LIBERO-Plus perturbation benchmark and produces a result table compatible with FastWAM's `summarize_results_plus.py`.

## Layout

| File | Role |
|---|---|
| `client.py` | Reads `suite,task_id` lines from a tasks file, runs 1 trial each via lingbot-va WebSocket server, writes per-task `_results.json`. Implements `libero_plus_smart` prompt logic locally so the eval has no FastWAM import dependency. |
| `launch_server.sh PORT GPU SAVE_ROOT CKPT_DIR` | Single-GPU server. Idempotently patches `va_libero_cfg.py` to point at `CKPT_DIR` and rewrites the ckpt's `transformer/config.json` `attn_mode: flex → torch` (required for inference per upstream README). |
| `launch_client.sh PORT TASKS_FILE OUTPUT_DIR` | Single-shard client. Reads `PROMPT_SOURCE`, `MAX_TIMESTEPS`, `SAVE_VIDEOS` from env. |
| `run_libero_plus_batch.sh` | 2-GPU (or N-GPU) sharded driver. Round-robins the task file, launches paired server+client per GPU on `BASE_PORT+i`, kills servers on exit. |

The slurm wrapper lives at [`scripts/eval_libero_plus_libero10_h200.slurm`](../../scripts/eval_libero_plus_libero10_h200.slurm).

## Output schema

`run_libero_plus_batch.sh` writes one JSON per task:

```
{OUTPUT_DIR}/{suite}/task{task_id}_results.json
{
  "task_suite": "libero_10",
  "task_id": 1234,
  "successes": 0|1,
  "total_episodes": 1,
  "prompt": "...",
  "bddl_file": "..."
}
```

This is the exact schema FastWAM's `summarize_results_plus.py` reads (recursive `*_results.json` glob).

## End-to-end on tillicum

```bash
# 1. one-time env setup (creates conda env + activate script)
bash scripts/setup_lingbot_va_tillicum.sh

# 2. submit the eval (downloads ckpt on first run, ~14B params)
sbatch scripts/eval_libero_plus_libero10_h200.slurm
```

Outputs land in `eval_results/lingbot_va_libero_long_libero_plus_libero10_<timestamp>/`:
- `shards/`     round-robin task splits
- `server_logs/`, `client_logs/`  per-GPU stdout
- `libero_10/task<N>_results.json`  per-task results
- `summary_plus.csv`, `summary_plus.json`  7-category × 4-suite summary (only libero_10 column populated)

## Smoke test (recommended before full run)

Run a 5-task subset first to verify the whole pipeline:

```bash
# write 5 task lines
head -5 logs/tasks_libero_plus_libero10.txt > /tmp/tasks_smoke.txt

OUTPUT_DIR=eval_results/smoke_$(date +%s)
NUM_GPUS=1 CKPT_DIR=$RUN_DIR/lingbot-va-posttrain-libero-long \
  TASK_FILE=/tmp/tasks_smoke.txt OUTPUT_DIR=$OUTPUT_DIR \
  SERVER_WARMUP_SEC=120 \
  bash evaluation/libero_plus/run_libero_plus_batch.sh
```

Then verify the summarizer accepts the output:

```bash
python $FASTWAM_ROOT/experiments/libero/summarize_results_plus.py \
  --output_dir $OUTPUT_DIR \
  --classification_json $LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json
```

## Notes / gotchas

- **`attn_mode` patch**: lingbot-va README warns the released ckpt ships with `attn_mode=flex` (training only). `launch_server.sh` rewrites it to `torch` on the fly. The patch is idempotent and only writes if currently `flex`.
- **Server warmup**: 14B params + VAE + T5 take ~60–120s to load. The batch script sleeps `SERVER_WARMUP_SEC` (default 60, slurm sets 120) before launching clients. Bump if you see `Connection refused` in client logs.
- **Resumability**: `client.py` skips any `(suite, task_id)` whose result JSON already exists, so a crashed/preempted run can be resumed by `sbatch`-ing the same script again with the same `OUTPUT_DIR`.
- **Init state**: only `init_states[0]` is used (1 trial/task), matching FastWAM's LIBERO-Plus convention.
- **Runtime estimate**: 2519 tasks × 1 trial × ~30–90s/episode ÷ 2 GPUs ≈ 10–30 wall-clock hours. Slurm allocates 24h.
- **Resolution**: 128×128 cameras, hardcoded to match `va_libero_cfg.py`. The release ckpt was trained at this resolution.
