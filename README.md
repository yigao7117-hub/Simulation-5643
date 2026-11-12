# Visual Distractors vs. Robot Manipulation Success

A study in Isaac Sim / Isaac Lab of how **visual distractors** affect an **image-based imitation-learning
policy** on a Franka cube-stacking task.

The policy only sees camera images plus proprioception, never object poses. We add distractor objects that change
the images but not the physics, and we compare success over **paired** episodes: the same initial states in every
condition.

We report three things:
- **Whether** distractors hurt.
- **Which property** of a distractor hurts.
- **How the policy fails**: which milestone it reaches, and how much inference it spends before failing.

![C0–C3 at step 200, same seed](media/grid_seed1000_step200.png)

*Seed 1000 at step 200, with each tile showing the table camera and the wrist camera. Top left is C0 (original), top
right C1 (+2 ordinary), bottom left C2 (+5 ordinary), bottom right C3 (+5 target-similar). Videos are in
[`media/`](media/).*

## Summary

- **Distractors cut success sharply, and more distractors cut it further.** Success falls from 79% with none to
  48% with 2 ordinary objects and 13% with 5 (100 paired episodes per condition, p < 0.001).
- **Cubes shaped like the targets but in ordinary colours are the most harmful distractor.** They bring success
  down to 3%. Target-like colours on ordinary shapes are no worse than ordinary distractors.
- **The policy does not compute slower; it gets lost.** One policy call takes about 5 ms either way. Failed
  episodes almost always time out after the first milestones, and **68–81% of their inference is spent after the
  last milestone was reached**.
- **The simulator is not deterministic.** Rerunning a seed gives the same outcome in 70–94% of cases, so every
  result is pooled over two independent runs.

## Project phases

| Phase | Question | Status |
|---|---|---|
| 0. Setup | Can we train an image-only policy in Isaac Lab 3.0 on an RTX 5090? | done |
| 1. Main experiment | Do distractors reduce success? Does the number of distractors matter? (C0–C3) | done (2 runs × 50 episodes) |
| 2. Distractor type | For similar distractors, does shape or colour matter? (C4, C5: 2×2 ablation) | done (2 runs × 50 episodes) |
| 2b. Failure and effort analysis | Where does the policy fail, and how much inference does it spend before failing? | done (run 2, instrumented) |
| 3. Mitigation | Does training with random distractors make the policy robust? | planned |
| 4. Second task | Do the findings transfer to another manipulation task? | planned |

## Phase 0: setup

| | |
|---|---|
| Task | `IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor` (Isaac Lab 3.0): stack red on blue, then green on red |
| Policy | robomimic BC-RNN-GMM, ResNet18 + SpatialSoftmax encoders |
| Observations | `table_cam` + `wrist_cam` RGB (200×200), `eef_pos`, `eef_quat`, `gripper_pos` (no object poses) |
| Training data | 1000 demos generated with Isaac Lab Mimic from the official Franka stack teleop dataset |
| Training | 600 epochs planned; interrupted at about epoch 480 |

The checkpoint was selected on C0 with 6 episodes each (`results/checkpoint_selection/`):

| Epoch | 200 | 300 | 400 | 480 |
|---|---|---|---|---|
| C0 success | 0/6 | 3/6 | 1/6 | 5/6 |

Epoch 480 is used throughout.

## Phases 1 and 2: conditions and success rates

All conditions use 5 distractors unless noted. C2–C5 share the same 5 positions and orientations, so they differ
only in shape and/or colour.

| Condition | Distractors | Shape | Colour |
|---|---|---|---|
| C0 | none | – | – |
| C1 | 2 ordinary objects | ordinary | ordinary |
| C2 | 5 ordinary objects | ordinary | ordinary |
| C3 | 5 target-similar cubes | cube, same size as targets | target-like (hue-shifted red/green/blue) |
| C4 | ablation | cube, same size as targets | ordinary |
| C5 | ablation | ordinary | target-like |

**Results.** Each condition has 100 episodes: seeds 1000–1049, each run twice (run 1 and run 2). Paired
comparisons match the same run and seed.

| Condition | Success | Rate (95% Wilson CI) | Run 1 | Run 2 | vs. C0 (paired, exact McNemar) |
|---|---|---|---|---|---|
| C0 | 79/100 | **0.79** (0.70–0.86) | 0.86 | 0.72 | – |
| C1 | 48/100 | **0.48** (0.39–0.58) | 0.46 | 0.50 | 41 lost, 10 gained, p < 0.001 |
| C2 | 13/100 | **0.13** (0.08–0.21) | 0.16 | 0.10 | 67 lost, 1 gained, p < 0.001 |
| C3 | 14/100 | **0.14** (0.09–0.22) | 0.10 | 0.18 | 68 lost, 3 gained, p < 0.001 |
| C4 | 3/100 | **0.03** (0.01–0.09) | 0.00 | 0.06 | 76 lost, 0 gained, p < 0.001 |
| C5 | 14/100 | **0.14** (0.09–0.22) | 0.16 | 0.12 | 68 lost, 3 gained, p < 0.001 |

<img src="media/success_rates.png" width="640">

**Shape × colour ablation (phase 2).** Success rates with 5 distractors:

| | ordinary colours | target-like colours |
|---|---|---|
| **ordinary shapes** | C2: 0.13 | C5: 0.14 |
| **cube shape** | C4: **0.03** | C3: 0.14 |

| Paired comparison | Change | Only first succeeds | Only second succeeds | p |
|---|---|---|---|---|
| C2 vs C4 | shape → cube | 12 | 2 | **0.013** |
| C2 vs C5 | colour → target-like | 9 | 10 | 1.0 |
| C5 vs C3 | shape → cube | 8 | 8 | 1.0 |
| C4 vs C3 | colour → target-like | 3 | 14 | **0.013** |

Interpretation:
- **Target-like colours alone change nothing** (C2 ≈ C5).
- **A target-sized cube hurts the most when its colour is ordinary** (C4). Colouring the same cubes in
  target-like hues brings success back to the level of ordinary distractors (C3).
- **One possible explanation** is that the policy relies on cube shape to find objects. Ordinary-coloured cubes
  (white, black, yellow, purple, orange) then compete with the targets, while hue-shifted cubes remain easier to
  tell apart from the pure red, green and blue targets. This explanation is not confirmed.

## Phase 2b: how the policy fails, and inference effort

These numbers are from run 2, which logs every policy call. They are in `results/combined/effort.csv`,
`failure_timing.csv` and `retries.csv`.

**Failure mode.** The table counts failed episodes in run 2, 50 episodes per condition. "Stuck after" is the last
milestone reached before failing.

| Condition | Failures | Timeout | Cube dropped | Stuck after stacking red | Stuck after grasping green | First wrong-object approach (mean step) |
|---|---|---|---|---|---|---|
| C0 | 14 | 14 | 0 | 4 | 9 | – |
| C1 | 25 | 22 | 3 | 10 | 13 | 450 |
| C2 | 45 | 40 | 5 | 15 | 29 | 392 |
| C3 | 41 | 38 | 3 | 19 | 17 | 350 |
| C4 | 47 | 44 | 3 | 16 | 27 | 288 |
| C5 | 44 | 41 | 3 | 16 | 26 | 407 |

The remaining few failures stopped earlier, at "nothing reached" or "red grasped".

**Milestone timing.** Mean step at which each milestone is first reached, over episodes that reach it:

| Condition | Grasp red | Stack red | Grasp green | Success (end) |
|---|---|---|---|---|
| C0 | 40 | 93 | 175 | 252 |
| C1 | 41 | 106 | 204 | 325 |
| C2 | 45 | 140 | 275 | 403 |
| C3 | 51 | 130 | 315 | 495 |
| C4 | 57 | 141 | 288 | 495 |
| C5 | 45 | 117 | 250 | 340 |

**Inference effort.** Timings use `torch.cuda.synchronize()` around each policy call. The first CUDA call is
excluded by a warm-up pass. The RTX 5090 was shared with other users while C2–C5 ran, so absolute latencies there
are inflated: env-step time rose from about 55 to 95 ms. The call counts and the shares are not affected.

| Condition | ms per policy call | Policy calls per success | Policy calls per failure | Inference s per failure | Share of failure inference after last milestone |
|---|---|---|---|---|---|
| C0 | 5.2 | 252 | 800 | 4.4 | **0.81** |
| C1 | 4.9 | 325 | 765 | 3.7 | **0.80** |
| C2 | 6.9 | 403 | 780 | 5.4 | **0.68** |
| C3 | 6.0 | 495 | 778 | 4.6 | **0.75** |
| C4 | 6.6 | 495 | 777 | 5.1 | **0.72** |
| C5 | 6.6 | 340 | 780 | 5.2 | **0.74** |

Findings:
- **Clutter does not make a policy call slower.** A call takes about 5 ms, and one simulation and rendering step
  takes 50–95 ms, so the policy is never the bottleneck.
- **Clutter makes the policy slower to make progress.** Every milestone is reached later. A successful episode
  needs about 2× as many policy calls with 5 similar cubes (495) as without distractors (252).
- **Failures are almost always timeouts after partial progress, not drops.** In run 2, 199 of 216 failures were
  timeouts. The robot stacks red, often grasps green, then stops making progress. About 530–650 policy calls, or
  68–81% of the inference in a failed episode, are spent after the last milestone.
- **More target-like distractors pull the gripper sooner.** The first wrong-object approach comes at step 288 with
  target-sized cubes (C4), against step 450 with 2 ordinary objects (C1).

**Aborted and retried runs.** These are all evaluation process launches in run 2.

| | Value |
|---|---|
| Episodes | 300 |
| Process launches | 25 |
| Restarts after a real cube drop (by design, see below) | 17 |
| Crashed launches (segfault, exit 139) | 2 |
| Episodes discarded as invalid | 0 |

Every lost episode was rerun automatically, and the logs are in `results/run2/eval_*/attempts.csv` and
`events.csv`. Run 1 predates this logging. In run 1, an evaluation bug was found and fixed (see Limitations), and
the affected episodes were rerun.

**Run-to-run agreement.** Same condition and seed, run 1 vs run 2:

| C0 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| 70% | 80% | 82% | 84% | 94% | 88% |

## Experimental design details

- **Paired initial states.** Episode *i* uses seed 1000+*i* in every condition, giving the same cube poses and
  joint noise. Cameras, lighting, textures, targets and the horizon (800 steps) are fixed.
- **Purely visual distractors.** Distractors are kinematic and collision-free
  ([`envs/distractors.py`](envs/distractors.py)), so they change the images only.
- **Placement.** Distractors sit in two side bands (x ∈ [0.35, 0.60] m, |y| ∈ [0.15, 0.22] m) next to the cube
  workspace (|y| ≤ 0.10 m), at least 8 cm apart. They are off the manipulation path and never occlude the targets.
  Their layout comes from a separate RNG seeded with the episode seed.
- **Ordinary set.** A yellow cylinder, purple sphere, orange cone, white capsule and black cylinder.
- **Target-similar set.** Target-sized cubes in orange-red, violet, teal, crimson and yellow-green. The targets
  remain the only pure red, blue and green cubes, so the task stays uniquely specified.
- **Ablation sets.** C4 and C5 recombine these shapes and colours slot by slot.
- **Warm-up.** Three no-op steps run after each reset, because camera frames right after a reset are stale in
  Isaac Lab 3.0.
- **Outcome labels.**

  | Label | Meaning |
  |---|---|
  | `success` | The `cubes_stacked` success term fired. |
  | `manipulation_drop` | A cube fell off the table. |
  | `wrong_object` | Heuristic: the gripper commanded closed with the end-effector below 8 cm and within 4 cm (xy) of a distractor. |
  | `timeout_fail_<stage>` | The episode timed out after reaching that stage. |
- **Per-episode instrumentation (run 2).**
  - First step of each milestone.
  - Last milestone reached and steps after it.
  - Per-call inference latency: mean, p50, p95 and max.
  - Inference time after the last milestone.
  - Env-step latency and wall time.
  - The process launch the episode ran in.
  - Per-step traces in `traces/*.npz`: latency, stage, and end-effector–distractor distance.

## Repository layout

```
configs/bc_rnn_image.json      robomimic config (Isaac Lab bc_rnn_image_200.json, num_data_workers 4 -> 12)
envs/distractors.py            distractor sets (ordinary, similar, shape_only, color_only), placement, reset event
scripts/
  smoke_test.py                launch the env, save camera frames (installation check)
  train.py                     wrapper around Isaac Lab's robomimic train.py using configs/bc_rnn_image.json
  eval_distractors.py          paired evaluation of one condition (CSV, summary, traces, videos, event log)
  run_eval_resilient.sh        reruns eval_distractors.py until done; logs every launch to attempts.csv
  run_all.sh                   all conditions sequentially (clean timing), then analysis and media
  run_when_idle.sh             waits until no other user is on the GPU, then runs run_all.sh
  analyze.py                   per-run analysis (rates, CIs, McNemar, failures, effort, retries)
  combine_runs.py              pooled analysis over repeated runs
  make_media.py                labelled 2x2 comparison videos into media/
results/
  combined/                    main results: run1 + run2 pooled (summary, paired tests, effort, timing, retries)
  run1/                        run 1: C0-C5 x 50 episodes
  run2/                        run 2: C0-C5 x 50 episodes, instrumented (traces/, attempts.csv, events.csv)
  checkpoint_selection/        C0 success of epochs 100-480
  layout_preview/              first frames of the C2-C5 layouts
media/                         comparison videos (run 1, seeds 1000/1001/1003), success-rate figure
```

Not tracked (see `.gitignore`):
- `data/`: about 21 GB of HDF5 datasets.
- `checkpoints/`: 140 MB each; the trained checkpoints are not released.
- `logs/`.
- Per-episode videos in `results/**/videos/`.

## Reproducing

### 1. Environment

Tested on Ubuntu 24.04, RTX 5090 (32 GB), driver 595.99, CUDA 13.2. The software stack is Isaac Sim 6.2.0 and
Isaac Lab v3.0.0, with Python 3.12, torch 2.11.0+cu130 and robomimic 0.4.0.

> Isaac Sim 5.1 crashed (`librtx.scenedb` segfault) on this Blackwell GPU with the 595 driver, so 6.2 is used.
> Isaac Lab 3.0 changed APIs: launch through `launch_simulation`, headless is the default, the task was renamed, and
> data is accessed through `.torch`. The scripts here target 3.0.

```bash
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git ~/isaac/IsaacLab
cd ~/isaac/IsaacLab
export UV_PROJECT_ENVIRONMENT=~/isaac/env_isaaclab OMNI_KIT_ACCEPT_EULA=YES
uv sync --extra isaacsim --extra mimic
source ~/isaac/env_isaaclab/bin/activate
export ISAACLAB_PATH=~/isaac/IsaacLab
```

The commands below assume this environment is active and are run from the repository root.

```bash
python scripts/smoke_test.py   # should save camera images and a summary to results/smoke_test/
```

### 2. Data

Download the official Franka stack teleop dataset (10 demos) from NVIDIA:
<https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5>.
Save it as `data/franka_stack_dataset.hdf5`, then:

```bash
M=$ISAACLAB_PATH/scripts/imitation_learning/isaaclab_mimic
python $M/annotate_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 --auto \
    --input_file data/franka_stack_dataset.hdf5 --output_file data/annotated_dataset.hdf5
python $M/generate_dataset.py --num_envs 100 --generation_num_trials 1000 \
    --input_file data/annotated_dataset.hdf5 --output_file data/generated_dataset_1000.hdf5
```

### 3. Training

The trained checkpoints are not released, so the policy has to be trained locally. The run below trains 600
epochs × 500 steps and saves a checkpoint every 20 epochs.

```bash
python scripts/train.py --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor --algo bc \
    --dataset data/generated_dataset_1000.hdf5 --log_dir $PWD/checkpoints/bc_rnn_image
```

### 4. Evaluation

```bash
# preview distractor layouts (no policy)
python scripts/eval_distractors.py --condition C4 --preview --num_episodes 2 --run_name layout_preview

# one condition
python scripts/eval_distractors.py --condition C2 --checkpoint <model_epoch_480.pth> --run_name run2

# all six conditions x 50 episodes, sequentially, then per-run analysis and media
scripts/run_all.sh <model_epoch_480.pth> run2 50

# pool repeated runs
python3 scripts/combine_runs.py results/combined results/run1 results/run2
```

## Limitations

- **Single source demo.** Only 1 of the 10 official demos replayed cleanly during annotation on Isaac Sim 6.2. All
  1000 generated demos therefore derive from that one source demo. This likely makes the policy more sensitive to
  distractors than a policy trained on diverse data.
- **One checkpoint.** Training was interrupted at about epoch 480 of 600. The checkpoint was chosen on seeds
  1000–1005, which overlap the evaluation seeds.
- **Non-deterministic physics.** GPU PhysX is not bit-exact. Pairing fixes the initial state but not the
  trajectory, so the same seed agrees across runs only 70–94% of the time. Results are pooled over two runs.
  Pooled pairs are not fully independent, because each seed appears twice.
- **Timing on a shared GPU.** Other users' jobs inflated the absolute latencies for C2–C5 in run 2.
- **Approximate wrong-object label.** `wrong_object` is a geometric heuristic, not a contact check.
- **Evaluation workaround.** After a real cube drop, the simulator could stay in a corrupted state that
  `env.reset()` did not clear, and later episodes "dropped" within 3 steps. The evaluation therefore restarts the
  process after every drop and resumes from `episodes.csv`. Drops within 5 steps are discarded and rerun. Both
  runs use this procedure.

## Acknowledgements and licenses

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause) provides the task, the Mimic data-generation
  pipeline and the robomimic training scripts and config.
- [Isaac Sim](https://developer.nvidia.com/isaac/sim) is subject to the NVIDIA Isaac Sim license and EULA.
- The Franka stack teleop dataset is an NVIDIA Isaac Lab asset and is not redistributed here.
- [robomimic](https://github.com/ARISE-Initiative/robomimic) (MIT) provides the BC-RNN-GMM algorithm and the visual
  encoders.
