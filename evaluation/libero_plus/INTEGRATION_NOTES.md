# LIBERO-Plus integration notes

Bring-up record for running `robbyant/lingbot-va-posttrain-libero-long` on
LIBERO-Plus on tillicum H200. Complements the quickstart in
[`README.md`](README.md). Read this doc when something breaks or before
porting the eval to a new cluster — it logs the fourteen bugs that surfaced
during integration and how each was resolved.

## TL;DR

```bash
# one-time
bash scripts/setup_lingbot_va_tillicum.sh

# per eval
sbatch scripts/eval_libero_plus_libero10_h200.slurm
```

Subset: 420 tasks (7 perturbation categories × 60 evenly-spaced task_ids),
filtered from FastWAM's `generate_subeval_tasks.py` output to libero_10 only.
Same selection as FastWAM sub-eval, so SR table is directly comparable.

Runtime estimate: ~9-12 GPU-hours on 2 H200, well under the 24h slurm cap.

## Architecture

```
slurm (eval_libero_plus_libero10_h200.slurm)
  ├─ source activate_lingbot_va_tillicum.sh    # conda env + cmake on PATH
  ├─ huggingface-cli download (idempotent)     # ckpt to runs/
  ├─ rm -rf ~/.libero ; python <<< "N"         # init LIBERO-plus config
  ├─ generate_subeval_tasks.py + grep libero_10 (idempotent)
  └─ run_libero_plus_batch.sh
       ├─ shard task file round-robin → 2 GPUs
       ├─ per GPU: launch_server.sh (background)
       │     ├─ patch va_libero_cfg.py with ckpt path (sed)
       │     ├─ patch ckpt's transformer/config.json: attn_mode flex→torch
       │     └─ python -m torch.distributed.run wan_va_server.py …
       ├─ wait SERVER_WARMUP_SEC (240)
       └─ per GPU: launch_client.sh (foreground, waits)
             └─ client.py
                  ├─ read tasks file (suite,task_id lines)
                  ├─ resolve_prompt() with libero_plus_smart logic
                  ├─ run_one_task() → env_one_step() with gripper fix
                  └─ write {OUTPUT_DIR}/{suite}/task{N}_results.json
  └─ summarize_results_plus.py → 7×4 SR table (libero_10 column populated)
```

## Files added/modified by this branch

| File | Type | What it does |
|---|---|---|
| `evaluation/libero_plus/client.py` | new | tasks-file driven client; libero_plus_smart prompt; **gripper sign-flip + threshold**; resumable (skips existing result JSON) |
| `evaluation/libero_plus/launch_server.sh` | new | per-GPU server; patches ckpt path + attn_mode; randomized master_port |
| `evaluation/libero_plus/launch_client.sh` | new | per-shard client wrapper |
| `evaluation/libero_plus/run_libero_plus_batch.sh` | new | shard task file across N GPUs, paired server/client |
| `evaluation/libero_plus/README.md` | new | quickstart |
| `evaluation/libero_plus/INTEGRATION_NOTES.md` | new | this doc |
| `scripts/setup_lingbot_va_tillicum.sh` | new | one-shot env: conda env, deps, Kitware cmake, missing LIBERO-plus runtime deps, version pin recovery; writes activate script |
| `scripts/eval_libero_plus_libero10_h200.slurm` | new | sbatch wrapper, sub-LIBERO-Plus libero_10 only |
| `wan_va/modules/model.py` | modified | flash-attn import made optional (deferred to `attn_mode='flashattn'` selection) |

## Output schema

Each task writes one JSON. `summarize_results_plus.py` consumes via recursive `*_results.json` glob.

```json
{OUTPUT_DIR}/libero_10/task<N>_results.json
{
  "task_suite": "libero_10",
  "task_id": <int>,
  "successes": 0|1,
  "total_episodes": 1,
  "prompt": "...",
  "bddl_file": "..."
}
```

`successes` and `total_episodes` are the load-bearing fields (read by the summarizer).

## Bugs discovered & fixes

Grouped by phase. Symptom → root cause → where fixed → how verified.

### Phase 1: env setup

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `from flash_attn ... import flash_attn_func` ImportError on `wan_va` import | Hard import at module load, but inference uses `attn_mode='torch'` (SDPA) which never calls `flash_attn_func` | `wan_va/modules/model.py`: wrap import in try/except, set `flash_attn_func=None` on miss; defer `ImportError` to `attn_mode='flashattn'` branch |
| 2 | `pip install flash-attn` fails: "nvcc not found, CUDA_HOME not set" on login node | Login node has CUDA runtime but no toolkit | Setup script skips flash-attn by default (set `INSTALL_FLASH_ATTN=1` to opt in). Inference works without it via fix #1 |
| 3 | `pip install` IncompleteRead 36977/37524 bytes — repeatable identical truncation | Cluster network/proxy interferes with pip's index/wheel fetch; cached partial download stuck | Add `--timeout 300 --retries 10 --no-cache-dir` to all pip calls; suggest `pip cache purge` if it persists |
| 4 | `MINIFORGE_ROOT` placeholder in setup didn't match user's actual conda install | Hardcoded path drift between users | Auto-detect from `command -v conda` (two `dirname` up); env override still possible |
| 5 | conda+pip cmake conflict: `pip uninstall cmake` removed conda's binary too | Both publish `cmake-4.3.2`; pip's metadata sees the conda-installed binary as "pip-managed" | Setup downloads Kitware's official prebuilt cmake-3.31.5 tarball into `~/local`, prepends to PATH; activate script propagates PATH |
| 6 | `LIBERO-plus/libero/libero/envs/env_wrapper.py` `ModuleNotFoundError: wand` (and later `skimage`, `imutils`) | These imports aren't declared in LIBERO-plus's `requirements.txt` | Setup explicitly installs `Wand` (Python) + `imagemagick` (C lib via conda) + `scikit-image` + `imutils` |
| 7 | `cv2` runtime: "module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000" | `scikit-image>=0.25` pulled `numpy 2.x`; `opencv-python==4.6.0.66` (from LIBERO-plus reqs) is numpy-1.x ABI | Setup pins `numpy<2` (1.26.4 latest 1.x; satisfies skimage `>=1.24` AND cv2's 1.x ABI) |
| 8 | `from diffusers import AutoencoderKLWan` fails: "cannot import name 'AutoImageProcessor' from 'transformers'" | LIBERO-plus reqs pinned `transformers==4.21.1`; `diffusers==0.36.0` needs ≥4.30 | Setup re-installs `transformers==4.55.2` and `tokenizers>=0.21` AFTER LIBERO-plus reqs to override the downgrade |

### Phase 2: server inference

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 9 | Server crash on second smoke run: `EADDRINUSE port: 29556` | torch.distributed master_port computed deterministically from PORT; previous run's TCPStore in TIME_WAIT | `launch_server.sh`: `MASTER_PORT="$((20000 + (PORT * 13 + $$) % 30000))"` — mix in PID for back-to-back collision avoidance |
| 10 | Server log: `'patch_embedding.bias, patch_embedding.weight' were not used when initializing` — looked like missing layer | Misleading warning. Ckpt has both legacy Conv3D `patch_embedding` AND lingbot-va's `patch_embedding_mlp` (Linear); model class only declares the latter, so the former is harmlessly unused | No fix needed. Confirmed via `safetensors.safe_open` + `WanTransformer3DModel.state_dict()` diff: `patch_embedding_mlp` IS loaded with matching shape `(3072, 192)` |

### Phase 3: client / eval glue

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 11 | Client crash: `EOFError: EOF when reading a line` at `from libero.libero import benchmark` | LIBERO's `__init__.py` calls `input(...)` on first run to set up `~/.libero/config.yaml`; client subshell has no stdin | Slurm preamble does `rm -rf ~/.libero` then `python -c "from libero.libero import benchmark" <<< "N"` to seed the config before any client subshell starts |
| 12 | Vanilla LIBERO sanity: `torch.load(init_states_path)` raises UnpicklingError on `numpy.core.multiarray._reconstruct` | PyTorch 2.6+ flipped default `weights_only=True`; vanilla LIBERO's `benchmark/__init__.py` doesn't pass the kwarg. LIBERO-plus had patched their fork already | One-line `sed -i` patch on the local LIBERO clone: `torch.load(init_states_path, weights_only=False)`. **NOT in slurm path** — slurm uses LIBERO-plus, not vanilla. Documented for sanity workflow only. |
| 13 | First sanity: 0/3 success on vanilla libero_10 (paper reports 98.5%) → suspected pipeline broken | **Gripper convention mismatch**. Two effects compounded: (a) lingbot-va's LeRobot training data has gripper sign reversed from LIBERO Panda env (training: `+=open/-=close`, env: `+=close/-=open`); (b) flow-matching diffusion outputs are soft (~±0.3-0.5), and LIBERO Panda controller is magnitude-sensitive (0.3 = partial close → object slips) | `client.py` before `env.step()`: `ee_action[6] = -1.0 if ee_action[6] > 0 else 1.0` (sign flip + hard threshold). Verified by trying all 4 (flip × threshold) combinations: only flip+threshold succeeds. Vanilla task0 went from `success=0` → `success=1` |
| 14 | LIBERO-plus benchmark requires perturbation BDDL prompts that the model never trained on (e.g. `LIVING_ROOM_SCENE2_..._table_1.bddl`) | `task.language` would leak `"table 1"` into the instruction; `bddl_language` would feed LIBERO-PRO's rewritten wording | `prompt_source=libero_plus_smart`: per-task hybrid — if filename contains `_language_` use the perturbed BDDL `:language` (Language Instructions category); otherwise strip `_table_/_light_/_view_/_initstate_/_noise_/...` suffixes back to canonical filename and derive training-set wording. Logic copied verbatim from FastWAM's `libero_utils.py` to keep zero cross-repo dep at runtime |

## Diagnostic dead-ends (didn't help)

Notes on hypotheses that turned out wrong, in case someone re-debugs:

- **Image vertical flip `[::-1]`**: suspected wrong direction → wasted ~30 min checking. The flip is correct (LIBERO's offscreen renderer returns upside-down images; `[::-1]` corrects). Confirmed by saving rollout video.
- **`patch_embedding` not loaded**: server's "not used" warning looked like a missing-layer bug. Spent time diffing ckpt keys vs model state_dict. Result: ckpt has redundant legacy keys; the loaded layer is `patch_embedding_mlp` and works fine.
- **`init_states[0]` particularly hard**: hypothesis was that we picked an unlucky init pose. Actually unrelated — the gripper bug masked all init states equally.
- **Action denormalization scale**: spent time analyzing `q01/q99` for EEF channels. They're correct (in [-1, 1] OSC normalized space, matching LIBERO Panda controller).

## Diagnostic that worked

- **Save and watch a rollout video**. Single most useful tool. From the video the user identified that EEF motion was sensible but gripper was failing — narrowed root cause to 1/7 action dimensions instead of guessing across 7 + image + sim.
- **Dump `actions_*.pt`** from server's save_async output to a Python REPL and inspect mean/range per channel across chunks. Made the "magnitude is soft" + "sign reversed for late chunks" pattern obvious.

## Open items / not-yet-fixed

| Item | Severity | Notes |
|---|---|---|
| Gripper jitter at `~0` boundary | low — visual artifact, doesn't always cost SR | Hard threshold flips sign whenever model output crosses 0; could add hysteresis (e.g. only flip when crossing ±0.2 with state memory). Worth trying if SR has room after baseline. |
| No `--requeue` on slurm | low — 24h jobs rarely preempted | client.py is already resumable (skips existing result JSON) so manual re-`sbatch` works |
| Only libero_10 sub eval (420 tasks) | by design — see goal | Full 4-suite sub would be 1680 tasks. The released ckpt is LIBERO-Long fine-tuned, so it's primarily compared on libero_10 anyway. |
| `evaluation/libero/client.py` (vanilla) is unchanged from upstream — gripper fix only in `evaluation/libero_plus/client.py` | low — vanilla eval is for sanity only | If we end up running vanilla LIBERO eval seriously, port the gripper fix there too |

## Verification log

| Step | Result |
|---|---|
| `python -c "import torch, diffusers; ..."` | torch 2.9.0+cu126, diffusers 0.36.0 |
| `python -c "from wan_va.modules.model import WanTransformer3DModel"` | Import OK without flash-attn |
| `from libero.libero import benchmark` (LIBERO-plus PYTHONPATH) | `libero_10 tasks: 2519` (vs vanilla's 10) |
| 5-task LIBERO-Plus smoke (pipeline) | 0/5 success but pipeline OK; result JSON schema valid; summarizer parses |
| Vanilla LIBERO task0 (no gripper fix) | 0/1 — caught the bug |
| Vanilla LIBERO task0 (gripper fix applied) | 1/1 ✓ — root cause confirmed |
