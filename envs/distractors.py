"""Visual distractors for the Franka cube-stacking visuomotor task.

Design (pilot study):
  * Distractors are kinematic and collision-free -> they change the *images* only, never the physics.
  * They are placed in two side bands outside the cube workspace (x in [0.4,0.6], y in [-0.1,0.1]),
    so they stay off the manipulation path and never sit between the table camera (x=1.0) and the cubes.
  * Placement uses a dedicated RNG seeded by the episode seed, so cube poses / joint noise (drawn from the
    global RNGs by the original reset events) are identical across conditions -> paired design.
  * "ordinary": non-cube shapes in non-target colors.
  * "shape_only" / "color_only": 2x2 ablation -- cubes in the ordinary colors / ordinary shapes in the similar colors.
    "similar":  5 cm cubes (same shape/size as targets) in hue-shifted colors; the targets remain the only
                pure red / blue / green cubes, so the task is still uniquely specified.
Unused slots are parked far outside every camera's view.
"""
from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

MAX_SLOTS = 5
CUBE_SIZE = 0.0468  # edge of the target blocks (matches cubes_stacked height_diff)
PARK_POS = (-3.0, 0.0, -0.5)  # out of view, below the table height

# Side bands (env frame): x range, |y| range. Targets occupy x[0.4,0.6], |y|<=0.1 (+ half cube).
BAND_X = (0.35, 0.60)
BAND_ABS_Y = (0.15, 0.22)  # wider |y| or larger x gets cut off at the table-camera edge
MIN_SEP = 0.08  # between distractors (center to center)
Z_ON_TABLE = 0.0

# (shape, sRGB color) per slot. Colors avoid the targets' pure red/green/blue.
# PreviewSurface diffuse_color is linear, so sRGB values are converted in _spawner().
ORDINARY = [
    ("cylinder", (0.95, 0.85, 0.10)),  # yellow cylinder
    ("sphere", (0.60, 0.20, 0.80)),    # purple sphere
    ("cone", (1.00, 0.55, 0.00)),      # orange cone
    ("capsule", (0.95, 0.95, 0.95)),   # white capsule
    ("cylinder", (0.10, 0.10, 0.10)),  # black cylinder
]
SIMILAR = [
    ("cube", (0.85, 0.30, 0.05)),  # orange-red  (near red target)
    ("cube", (0.40, 0.10, 0.70)),  # violet      (near blue target)
    ("cube", (0.00, 0.55, 0.50)),  # teal        (near green/blue)
    ("cube", (0.75, 0.05, 0.40)),  # crimson-magenta (near red)
    ("cube", (0.45, 0.70, 0.05)),  # yellow-green (near green target)
]

# 2x2 ablation (shape x color), same slot order as above:
SHAPE_ONLY = [("cube", rgb) for _, rgb in ORDINARY]                            # target-like shape, ordinary colors
COLOR_ONLY = [(shape, rgb) for (shape, _), (_, rgb) in zip(ORDINARY, SIMILAR)]  # ordinary shapes, target-like colors


def _srgb_to_linear(rgb):
    return tuple(float(c) ** 2.2 for c in rgb)


def _spawner(shape: str, rgb):
    rgb = _srgb_to_linear(rgb)
    common = dict(
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5),
    )
    s = CUBE_SIZE
    if shape == "cube":
        return sim_utils.CuboidCfg(size=(s, s, s), **common), s / 2
    if shape == "sphere":
        return sim_utils.SphereCfg(radius=0.025, **common), 0.025
    if shape == "cylinder":
        return sim_utils.CylinderCfg(radius=0.022, height=0.07, **common), 0.035
    if shape == "cone":
        return sim_utils.ConeCfg(radius=0.03, height=0.07, **common), 0.035
    if shape == "capsule":
        return sim_utils.CapsuleCfg(radius=0.018, height=0.05, axis="X", **common), 0.018
    raise ValueError(shape)


def sample_distractor_xy(n: int, rng: np.random.Generator, max_tries: int = 2000) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(max_tries):
            x = rng.uniform(*BAND_X)
            y = rng.uniform(*BAND_ABS_Y) * rng.choice([-1.0, 1.0])
            if all(np.hypot(x - px, y - py) >= MIN_SEP for px, py in out):
                out.append((x, y))
                break
        else:
            raise RuntimeError("could not place distractors without overlap")
    return out


def place_distractors(env, env_ids, n_active: int, half_heights: list[float], base_seed_attr: str = "_distractor_seed"):
    """Reset event: put the first n_active slots on the table, park the rest."""
    if env_ids is None:
        return
    seed = getattr(env, base_seed_attr, 0)
    rng = np.random.default_rng(seed)
    xy = sample_distractor_xy(n_active, rng)
    yaws = rng.uniform(-np.pi, np.pi, size=MAX_SLOTS)
    placed = []
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.arange(env.num_envs, device=env.device)
    for i in range(MAX_SLOTS):
        obj = env.scene[f"distractor_{i}"]
        if i < n_active:
            pos = (xy[i][0], xy[i][1], Z_ON_TABLE + half_heights[i])
        else:
            pos = PARK_POS
        yaw = float(yaws[i])
        quat = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))  # (w, x, y, z)
        for e in ids.tolist():
            p = torch.tensor([pos], device=env.device, dtype=torch.float32) + env.scene.env_origins[e, 0:3].float()
            q = torch.tensor([quat], device=env.device, dtype=torch.float32)
            obj.write_root_pose_to_sim_index(
                root_pose=torch.cat([p, q], dim=-1), env_ids=torch.tensor([e], device=env.device)
            )
        if i < n_active:
            placed.append({"slot": i, "x": round(pos[0], 4), "y": round(pos[1], 4)})
    env._distractor_layout = placed


def add_distractors(env_cfg, kind: str, n_active: int):
    """Mutate a FrankaCubeStackVisuomotorEnvCfg in place (call before launch)."""
    assert 0 <= n_active <= MAX_SLOTS
    palette = {"ordinary": ORDINARY, "similar": SIMILAR, "shape_only": SHAPE_ONLY, "color_only": COLOR_ONLY}[kind]
    half_heights = []
    for i in range(MAX_SLOTS):
        shape, rgb = palette[i]
        spawn, hh = _spawner(shape, rgb)
        half_heights.append(hh)
        setattr(
            env_cfg.scene,
            f"distractor_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Distractor_{i}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=PARK_POS),
                spawn=spawn,
            ),
        )
    # Runs after the original events (joint noise, cube poses), uses its own RNG.
    env_cfg.events.place_distractors = EventTerm(
        func=place_distractors, mode="reset", params={"n_active": n_active, "half_heights": half_heights}
    )
    return env_cfg


def distractor_positions(env) -> list[tuple[float, float]]:
    return [(d["x"], d["y"]) for d in getattr(env, "_distractor_layout", [])]
