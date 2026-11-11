"""Smoke test: launch the Franka stack-cube visuomotor env headless, step it, save camera frames.

Run:  source ~/isaac/activate.sh && python scripts/smoke_test.py --headless
"""
import argparse
import json
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "smoke_test"))
add_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)


def run(cfg):
    env = gym.make(args.task, cfg=cfg)
    t0 = time.time()
    obs, _ = env.reset(seed=0)
    for _ in range(args.steps):
        act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, *_ = env.step(act)
    summary = {"task": args.task, "steps": args.steps, "seconds": round(time.time() - t0, 2),
               "action_shape": list(env.action_space.shape), "obs": {}}
    for group, d in obs.items():
        if not isinstance(d, dict):
            summary["obs"][group] = list(d.shape)
            continue
        for k, v in d.items():
            summary["obs"][f"{group}/{k}"] = list(v.shape)
            if v.ndim == 4 and v.shape[-1] in (3, 4):  # (N,H,W,C) image
                Image.fromarray(to_uint8(v[0, ..., :3].detach().cpu().numpy())).save(
                    os.path.join(args.out, f"{group}_{k}.png"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SMOKE_OK", json.dumps(summary))
    env.close()


if __name__ == "__main__":
    os.makedirs(args.out, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, overrides=hydra_overrides)
    cfg.observations.policy.concatenate_terms = False
    with launch_simulation(cfg, args):
        run(cfg)
