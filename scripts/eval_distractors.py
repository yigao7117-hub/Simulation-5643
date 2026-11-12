"""Paired evaluation of a robomimic visuomotor policy under visual-distractor conditions.

One process = one condition (scene layout differs per condition).  Episode i uses seed (seed_base + i)
in every condition, so cube poses / joint noise are identical across conditions (paired design).

Examples:
  source ~/isaac/activate.sh
  # preview layouts only (no policy, saves first frames):
  python scripts/eval_distractors.py --condition C3 --preview --num_episodes 3
  # evaluate:
  python scripts/eval_distractors.py --condition C2 --checkpoint checkpoints/.../model_epoch_480.pth --run_name pilot
"""
import argparse
import copy
import csv
import json
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT)

CONDITIONS = {
    "C0": {"kind": "ordinary", "n": 0, "desc": "original scene"},
    "C1": {"kind": "ordinary", "n": 2, "desc": "+2 ordinary distractors"},
    "C2": {"kind": "ordinary", "n": 5, "desc": "+5 ordinary distractors"},
    "C3": {"kind": "similar", "n": 5, "desc": "+5 target-similar distractors"},
    # ablation: which property of the similar distractors matters?
    "C4": {"kind": "shape_only", "n": 5, "desc": "+5 cubes in ordinary colors (shape only)"},
    "C5": {"kind": "color_only", "n": 5, "desc": "+5 ordinary shapes in target-like colors (color only)"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor")
parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--seed_base", type=int, default=1000)
parser.add_argument("--horizon", type=int, default=800, help="max env steps per episode (same as Isaac Lab play.py)")
parser.add_argument("--run_name", default=time.strftime("%Y%m%d_%H%M"))
parser.add_argument("--preview", action="store_true", help="no policy: save first frame of each episode only")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--debug_force_drop", action="store_true", help="debug: push cube_1 below the table in episode 0")
add_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.envs.utils._manual_success import _prepare_success_term  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from envs.distractors import add_distractors, distractor_positions  # noqa: E402

# wrong-object heuristic: gripper commanded closed, EE low and within this xy radius of a distractor
WRONG_XY = 0.04
WRONG_Z = 0.08
WARMUP_STEPS = 3
INVALID_DROP_STEPS = 5
STAGES = ["none", "grasped_red", "stacked_red", "grasped_green"]  # subtask_terms progression
MILESTONES = ["grasp_1", "stack_1", "grasp_2"]  # -> grasped_red, stacked_red, grasped_green
ATTEMPT = int(os.environ.get("EVAL_ATTEMPT", "1"))  # process launch number, set by run_eval_resilient.sh


def log_event(out_dir, event, episode="", seed="", detail=""):
    """Append-only run log: process starts, discarded episodes, restarts (survives crashes)."""
    path = os.path.join(out_dir, "events.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "condition", "attempt", "event", "episode", "seed", "detail"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.condition, ATTEMPT, event, episode, seed, detail])


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def frame(obs_dict):
    t = obs_dict["policy"]["table_cam"][0].cpu().numpy().astype(np.uint8)
    w = obs_dict["policy"]["wrist_cam"][0].cpu().numpy().astype(np.uint8)
    return np.concatenate([t, w], axis=1)


def policy_obs(obs_dict, image_keys):
    obs = copy.deepcopy(obs_dict["policy"])
    for k in obs:
        obs[k] = torch.squeeze(obs[k])
    for k in image_keys:
        if k in obs_dict["policy"]:
            img = torch.squeeze(obs_dict["policy"][k]).permute(2, 0, 1).clone().float() / 255.0
            obs[k] = img.clip(0.0, 1.0)
    return obs


def run_episode(env, policy, success_term, ep, out_dir, image_keys):
    seed = args.seed_base + ep
    env._distractor_seed = seed
    env.seed(seed)
    obs_dict, _ = env.reset()
    # Warm-up: camera obs right after reset are stale in this Isaac Lab version. A few no-op steps
    # (zero IK delta = hold pose, gripper open) refresh the sensors; identical in every condition.
    noop = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    for _ in range(WARMUP_STEPS):
        obs_dict, *_ = env.step(noop)
    dpos = distractor_positions(env)
    frames = [frame(obs_dict)]
    if args.preview:
        Image.fromarray(frames[0]).save(os.path.join(out_dir, f"preview_ep{ep:02d}_seed{seed}.png"))
        return {"episode": ep, "seed": seed, "distractors": json.dumps(dpos)}

    policy.start_episode()
    if args.debug_force_drop and ep == 0:
        c1 = env.scene["cube_1"]
        pose = c1.data.root_pose_w.torch[0:1].clone().float()
        pose[0, 2] = -0.5
        c1.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=env.device))
    stage, wrong_steps, min_d = 0, 0, float("inf")
    outcome, steps = "timeout", args.horizon
    first_step = {m: None for m in MILESTONES}  # first policy step at which each milestone was reached
    first_wrong = None
    infer_ms, envstep_ms, stage_trace, dist_trace = [], [], [], []
    t_ep = time.perf_counter()
    for t in range(args.horizon):
        _sync()
        t0 = time.perf_counter()
        act = policy(policy_obs(obs_dict, image_keys))
        _sync()
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        act_t = torch.from_numpy(act).to(env.device).view(1, -1)
        ee = obs_dict["policy"]["eef_pos"][0].cpu().numpy()
        if dpos:
            d = min(np.hypot(ee[0] - x, ee[1] - y) for x, y in dpos)
            min_d = min(min_d, d)
            dist_trace.append(d)
            if act[-1] < 0 and ee[2] < WRONG_Z and d < WRONG_XY:
                wrong_steps += 1
                if first_wrong is None:
                    first_wrong = t + 1
        t1 = time.perf_counter()
        obs_dict, _, terminated, truncated, _ = env.step(act_t)
        envstep_ms.append((time.perf_counter() - t1) * 1000.0)
        sub = obs_dict["subtask_terms"]
        for s_idx, key in enumerate(MILESTONES, start=1):
            if bool(sub[key][0]):
                if first_step[key] is None:
                    first_step[key] = t + 1
                if stage < s_idx:
                    stage = s_idx
        stage_trace.append(stage)
        if not args.no_video:
            frames.append(frame(obs_dict))
        if bool(success_term.func(env, **success_term.params)[0]):
            outcome, steps = "success", t + 1
            break
        if bool(terminated[0]) or bool(truncated[0]):
            outcome, steps = "cube_dropped", t + 1
            tm = env.termination_manager
            fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
            cz = {k: round(float(env.scene[k].data.root_pos_w.torch[0, 2]), 3) for k in ("cube_1", "cube_2", "cube_3")}
            print(f"[TERM] seed={seed} t={t} fired={fired} cube_z={cz}", flush=True)
            break

    if outcome == "success":
        failure = ""
    elif outcome == "cube_dropped":
        failure = "manipulation_drop"
    elif wrong_steps > 0:
        failure = "wrong_object"
    else:
        failure = {0: "fail_grasp_red", 1: "fail_stack_red", 2: "fail_grasp_green", 3: "fail_stack_green"}[stage]
        failure = "timeout_" + failure

    wall_s = time.perf_counter() - t_ep
    if not args.no_video:
        import imageio

        imageio.mimsave(os.path.join(out_dir, "videos", f"ep{ep:02d}_seed{seed}_{outcome}.mp4"), frames, fps=20)

    # per-step trace (inference / env-step latency, milestone stage, EE-distractor distance)
    np.savez_compressed(
        os.path.join(out_dir, "traces", f"ep{ep:02d}_seed{seed}.npz"),
        infer_ms=np.array(infer_ms, np.float32), envstep_ms=np.array(envstep_ms, np.float32),
        stage=np.array(stage_trace, np.int8), ee_distractor_xy=np.array(dist_trace, np.float32),
    )
    inf = np.array(infer_ms)
    reached = [first_step[m] for m in MILESTONES if first_step[m] is not None]
    last_ms = max(reached) if reached else 0  # step of the last milestone reached (0 = none)
    after = inf[last_ms:]  # policy calls spent after the last milestone (unproductive for failures)
    return {
        "episode": ep, "seed": seed, "outcome": outcome, "success": int(outcome == "success"), "steps": steps,
        "max_stage": STAGES[stage], "failure_type": failure, "wrong_object_steps": wrong_steps,
        "min_ee_distractor_xy": round(min_d, 4) if dpos else "", "distractors": json.dumps(dpos),
        "step_grasp_red": first_step["grasp_1"] or "", "step_stack_red": first_step["stack_1"] or "",
        "step_grasp_green": first_step["grasp_2"] or "", "step_first_wrong_object": first_wrong or "",
        "last_milestone_step": last_ms, "steps_after_last_milestone": steps - last_ms,
        "policy_calls": len(inf), "infer_ms_total": round(float(inf.sum()), 1),
        "infer_ms_mean": round(float(inf.mean()), 3), "infer_ms_p50": round(float(np.percentile(inf, 50)), 3),
        "infer_ms_p95": round(float(np.percentile(inf, 95)), 3), "infer_ms_max": round(float(inf.max()), 3),
        "infer_ms_after_last_milestone": round(float(after.sum()), 1),
        "envstep_ms_mean": round(float(np.mean(envstep_ms)), 3), "episode_wall_s": round(wall_s, 2),
        "attempt": ATTEMPT,
    }


def main():
    cond = CONDITIONS[args.condition]
    sub = "preview" if args.preview else "eval"
    out_dir = os.path.join(PROJECT, "results", args.run_name, f"{sub}_{args.condition}")
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    if not args.preview:
        os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, overrides=hydra_overrides)
    cfg.observations.policy.concatenate_terms = False
    cfg.terminations.time_out = None
    cfg.recorders = None
    cfg.num_rerenders_on_reset = 3  # otherwise the first frame after reset is stale (pre-reset scene)
    success_term = _prepare_success_term(cfg)
    add_distractors(cfg, cond["kind"], cond["n"])
    image_keys = getattr(cfg, "image_obs_list", ["table_cam", "wrist_cam"])

    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        policy = None
        if not args.preview:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils

            device = TorchUtils.get_torch_device(try_to_use_cuda=True)
            policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device)
            # Timing hygiene: one throw-away forward pass so CUDA/cuDNN initialisation is not charged to
            # episode 0 (the RNN state is reset by start_episode() at the start of every episode).
            policy.start_episode()
            with torch.inference_mode():
                policy(policy_obs(env.reset()[0], image_keys))
            log_event(out_dir, "process_start", detail=f"gpu={torch.cuda.get_device_name(0)}")
        rows = []
        csv_path = os.path.join(out_dir, "episodes.csv")
        if not args.preview and os.path.exists(csv_path):  # resume after a crash: keep finished episodes
            with open(csv_path) as fh:
                rows = [dict(r, success=int(r["success"])) for r in csv.DictReader(fh)
                        if int(r["seed"]) == args.seed_base + int(r["episode"])]
            print(f"[RESUME] {len(rows)} episodes already done", flush=True)
        done = {int(r["episode"]) for r in rows}
        with torch.inference_mode():
            for ep in range(args.num_episodes):
                if ep in done:
                    continue
                r = run_episode(env, policy, success_term, ep, out_dir, image_keys)
                r["condition"] = args.condition
                if r.get("outcome") == "cube_dropped" and r["steps"] <= INVALID_DROP_STEPS:
                    # The sim was left corrupted by an earlier blow-up: a cube "drops" right after reset.
                    # Discard and restart the process (run_eval_resilient.sh resumes from episodes.csv).
                    print(f"[INVALID] ep{ep} dropped at step {r['steps']} -> restarting process", flush=True)
                    log_event(out_dir, "episode_discarded_invalid_drop", ep, r["seed"], f"dropped at step {r['steps']}")
                    break
                rows.append(r)
                print(f"[EVAL] {args.condition} ep{ep} {r.get('outcome', 'preview')} {r.get('failure_type', '')}", flush=True)
                write_outputs(rows, out_dir, cond)  # incremental: survives crashes / app shutdown
                if r.get("outcome") == "cube_dropped":
                    # A real drop can leave PhysX in a bad state that reset() does not clear: start a fresh process.
                    print("[RESTART] after drop", flush=True)
                    log_event(out_dir, "restart_after_drop", ep, r["seed"], f"drop at step {r['steps']}")
                    break
        env.close()


def write_outputs(rows, out_dir, cond):
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if not args.preview:
        n_succ = sum(r["success"] for r in rows)
        fails = {}
        for r in rows:
            if r["failure_type"]:
                fails[r["failure_type"]] = fails.get(r["failure_type"], 0) + 1
        summary = {
            "condition": args.condition, **cond, "episodes": len(rows), "successes": n_succ,
            "success_rate": n_succ / len(rows), "failure_types": fails, "checkpoint": args.checkpoint,
            "seed_base": args.seed_base, "horizon": args.horizon,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("[SUMMARY]", json.dumps(summary))


if __name__ == "__main__":
    main()
