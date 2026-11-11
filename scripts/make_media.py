"""Build viewer-friendly videos in media/ from eval runs.

For a given run and seed, tiles the C0..C3 episode videos (table_cam | wrist_cam) into a labeled 2x2 grid,
upscaled 2x, and copies the individual clips as well.

Usage: python scripts/make_media.py --run <run_name> --seed 1001 [--tag e100]
"""
import argparse
import glob
import os
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESC = {"C0": "C0 original", "C1": "C1 +2 ordinary", "C2": "C2 +5 ordinary", "C3": "C3 +5 similar"}

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--tag", default="")
p.add_argument("--scale", type=int, default=2)
a = p.parse_args()

media = os.path.join(PROJECT, "media")
os.makedirs(media, exist_ok=True)
tag = f"_{a.tag}" if a.tag else ""

clips = {}
for c in ["C0", "C1", "C2", "C3"]:
    hits = glob.glob(os.path.join(PROJECT, "results", a.run, f"eval_{c}", "videos", f"ep*_seed{a.seed}_*.mp4"))
    if hits:
        clips[c] = hits[0]
        outcome = hits[0].rsplit("_", 1)[-1].removesuffix(".mp4")
        shutil.copy(hits[0], os.path.join(media, f"{a.run}_seed{a.seed}_{c}_{outcome}{tag}.mp4"))
if not clips:
    raise SystemExit("no clips found")


def read(path):
    return [np.asarray(f) for f in imageio.mimread(path, memtest=False)]


frames = {c: read(p_) for c, p_ in clips.items()}
outcomes = {c: clips[c].rsplit("_", 1)[-1].removesuffix(".mp4") for c in clips}
T = max(len(v) for v in frames.values())
h, w = next(iter(frames.values()))[0].shape[:2]
blank = np.zeros((h, w, 3), np.uint8)

out = []
for t in range(T):
    tiles = []
    for c in ["C0", "C1", "C2", "C3"]:
        f = frames.get(c)
        img = blank if f is None else f[min(t, len(f) - 1)]
        im = Image.fromarray(img).resize((w * a.scale, h * a.scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        done = f is not None and t >= len(f) - 1
        label = DESC[c] + (f"  [{outcomes[c]}]" if done else "") if f is not None else DESC[c] + " (missing)"
        d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
        d.text((6, 5), f"{label}   step {min(t, len(f) - 1) if f else 0}", fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    top = np.concatenate(tiles[:2], axis=1)
    bot = np.concatenate(tiles[2:], axis=1)
    out.append(np.concatenate([top, bot], axis=0))

dst = os.path.join(media, f"{a.run}_seed{a.seed}_grid{tag}.mp4")
imageio.mimsave(dst, out, fps=20, macro_block_size=1)
print(dst)
