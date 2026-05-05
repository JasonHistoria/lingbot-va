"""LIBERO-Plus eval client for lingbot-va.

Rollout loop is copied verbatim from evaluation/libero/client.py (the vanilla
LIBERO-Long client that ships with the release ckpt). Only the task source,
prompt resolution, and output JSON schema are swapped:

  - Task source:  --tasks-file with `suite,task_id` lines (FastWAM format).
                  Lets one client process traverse heterogeneous suites
                  (e.g. only LIBERO-Plus's libero_10 perturbation tasks).
  - Prompt:       --prompt-source libero_plus_smart (default). Mirrors the
                  hybrid logic in FastWAM/experiments/libero/libero_utils.py
                  so the model receives the wording it actually saw during
                  training instead of LIBERO-Plus's perturbation tags.
  - Output:       per-task `{output_dir}/{suite}/task{task_id}_results.json`
                  with schema {task_suite, task_id, successes, total_episodes}
                  so FastWAM's summarize_results_plus.py can consume it
                  without modification.

Each task runs exactly 1 trial (init_states[0]); FastWAM's slurm uses the
same convention for LIBERO-Plus.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
    WebsocketClientPolicy,
)


# Regex from FastWAM/experiments/libero/libero_utils.py — strips LIBERO-Plus
# non-language perturbation suffixes back to canonical base filename.
# `_language_` is intentionally excluded.
_LIBERO_PLUS_NONLANG_SUFFIX_RE = re.compile(
    r"(_table_|_light_|_add_|_remove_|_view_|_initstate_|_noise_)\d.*\.bddl$"
)


def _resolve_bddl_for_language(task_bddl_file):
    """LIBERO-Plus _view_..._initstate_... suffixes have no separate BDDL —
    they reuse the base BDDL. env_wrapper strips at runtime, but we need
    the base path BEFORE env creation to read :language."""
    s = str(task_bddl_file)
    if "_view_" in s and "_initstate_" in s:
        return s.split("_view_")[0] + ".bddl"
    return s


def resolve_prompt(task, prompt_source):
    """Compute the task description string per `prompt_source`.

    Supported: task.language | bddl_language | libero_plus_smart.
    For LIBERO-Plus eval use libero_plus_smart.
    """
    bddl_path = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )

    if prompt_source == "task.language":
        return task.language

    if prompt_source == "bddl_language":
        parsed = BDDLUtils.robosuite_parse_problem(_resolve_bddl_for_language(bddl_path))
        li = parsed["language_instruction"]
        return " ".join(li) if isinstance(li, list) else li

    if prompt_source == "libero_plus_smart":
        bddl_filename = task.bddl_file
        if "_language_" in bddl_filename:
            parsed = BDDLUtils.robosuite_parse_problem(_resolve_bddl_for_language(bddl_path))
            li = parsed["language_instruction"]
            return " ".join(li) if isinstance(li, list) else li
        base = _LIBERO_PLUS_NONLANG_SUFFIX_RE.sub(".bddl", bddl_filename)
        stem = base[: -len(".bddl")] if base.endswith(".bddl") else base
        return stem.replace("_", " ")

    raise ValueError(f"unknown prompt_source: {prompt_source!r}")


def _read_tasks_file(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            suite, tid = line.split(",")
            out.append((suite.strip(), int(tid.strip())))
    return out


# ─── env helpers (copied verbatim from evaluation/libero/client.py) ──────────


def construct_single_env(env_args):
    count = 0
    env = None
    while count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            return env
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    return None


def _extract_obs(obs):
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.0] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def save_video(real_obs_list, save_path, fps=15,
               video_names=("observation.images.agentview_rgb",
                            "observation.images.eye_in_hand_rgb")):
    if not real_obs_list:
        return
    h, w = real_obs_list[0][video_names[0]].shape[:2]
    target = (w, h)
    frames = [
        np.hstack([cv2.resize(obs[name], target) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]
    imageio.mimsave(str(save_path), frames, fps=fps)


# ─── one task rollout (rollout body copied verbatim from libero client) ──────


def run_one_task(model, task, init_state, prompt, max_timesteps, save_video_path=None):
    env_args = {
        "bddl_file_name": str(
            pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = construct_single_env(env_args)
    if env is None:
        return False, []
    first_obs = init_single_env(env, init_state)

    model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    while env.env.timestep < max_timesteps:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret["action"]
        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j].copy()
                # Gripper convention fix:
                #   1. Sign flip: lingbot-va's LIBERO LeRobot training data has gripper
                #      sign reversed from LIBERO Panda env (training: +=open/-=close,
                #      env: +=close/-=open). Empirically verified by trying all 4 combos.
                #   2. Threshold to ±1: flow-matching diffusion output is soft (~±0.3-0.5)
                #      which only triggers partial close/open in LIBERO; we threshold
                #      to extremes for full grip/release. Without this gripper holds
                #      too weakly to lift objects.
                ee_action[6] = -1.0 if ee_action[6] > 0 else 1.0
                observes, done = env_one_step(env, ee_action)
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)
            if done:
                break
        first = False
        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True,
                             imagine=False, state=action))

    if save_video_path is not None:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(full_obs_list, save_video_path, fps=60)

    env.close()
    return bool(done), full_obs_list


# ─── main loop ───────────────────────────────────────────────────────────────


def run(tasks_file, port, output_dir, prompt_source, max_timesteps,
        save_videos):
    tasks = _read_tasks_file(tasks_file)
    print(f"[client] {tasks_file}: {len(tasks)} tasks; prompt_source={prompt_source}")

    model = WebsocketClientPolicy(port=port)

    # Cache one benchmark instance per suite so we don't reinstantiate per task.
    bench_cache = {}

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_succ = 0
    pbar = tqdm(tasks, total=len(tasks))
    for suite, task_id in pbar:
        result_file = out_root / suite / f"task{task_id}_results.json"
        if result_file.exists():
            try:
                prev = json.loads(result_file.read_text())
                n_succ += int(prev.get("successes", 0))
                n_done += int(prev.get("total_episodes", 0))
                pbar.set_postfix(succ=f"{n_succ}/{n_done}", suite=suite, task=task_id, skip=1)
                continue
            except Exception:
                pass  # malformed → re-run

        if suite not in bench_cache:
            bench_cache[suite] = benchmark.get_benchmark_dict()[suite]()
        bench = bench_cache[suite]
        if task_id >= bench.get_num_tasks():
            print(f"[client] WARN: {suite} task_id={task_id} ≥ num_tasks={bench.get_num_tasks()}, skipping")
            continue
        task = bench.get_task(task_id)
        init_states = bench.get_task_init_states(task_id)
        if init_states.shape[0] == 0:
            print(f"[client] WARN: {suite} task_id={task_id} has no init_states, skipping")
            continue

        prompt = resolve_prompt(task, prompt_source)

        video_path = (
            out_root / suite / "videos" / f"task{task_id}.mp4" if save_videos else None
        )
        try:
            success, _ = run_one_task(
                model, task,
                init_states[0],
                prompt,
                max_timesteps=max_timesteps,
                save_video_path=video_path,
            )
        except Exception as e:
            print(f"[client] ERROR rollout {suite}/{task_id}: {e}")
            success = False

        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps({
            "task_suite": suite,
            "task_id": task_id,
            "successes": int(bool(success)),
            "total_episodes": 1,
            "prompt": prompt,
            "bddl_file": task.bddl_file,
        }, indent=2))

        n_done += 1
        n_succ += int(bool(success))
        pbar.set_postfix(succ=f"{n_succ}/{n_done}", suite=suite, task=task_id)

    print(f"[client] DONE {tasks_file}: {n_succ}/{n_done} ({100.*n_succ/max(n_done,1):.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-file", type=str, required=True,
                   help="Tasks file with `suite,task_id` per line (FastWAM format)")
    p.add_argument("--port", type=int, required=True,
                   help="WebSocket port of the lingbot-va server")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Per-task results JSON dir; results land in {output_dir}/{suite}/task{task_id}_results.json")
    p.add_argument("--prompt-source", type=str, default="libero_plus_smart",
                   choices=["task.language", "bddl_language", "libero_plus_smart"])
    p.add_argument("--max-timesteps", type=int, default=800,
                   help="Per-episode timestep cap (matches lingbot-va's vanilla LIBERO client)")
    p.add_argument("--save-videos", action="store_true",
                   help="Save per-task rollout MP4 (off by default to save disk on 2.5k-task sweeps)")
    args = p.parse_args()
    run(args.tasks_file, args.port, args.output_dir, args.prompt_source,
        args.max_timesteps, args.save_videos)


if __name__ == "__main__":
    main()
