"""Run Isaac Lab's robomimic train.py with the project config (configs/bc_rnn_image.json).

Only change vs. the official bc_rnn_image_200.json: num_data_workers 4 -> 12 (training was data-loading bound).
Usage: python scripts/train.py --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor --algo bc \
           --dataset data/generated_dataset_1000.hdf5 --log_dir $PWD/checkpoints/bc_rnn_image
"""
import os
import runpy
import sys

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  (registers tasks)

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASK = sys.argv[sys.argv.index("--task") + 1]
gym.spec(TASK).kwargs["robomimic_bc_cfg_entry_point"] = os.path.join(PROJECT, "configs", "bc_rnn_image.json")

TRAIN = os.path.join(os.environ["ISAACLAB_PATH"], "scripts", "imitation_learning", "robomimic", "train.py")
sys.argv[0] = TRAIN
runpy.run_path(TRAIN, run_name="__main__")
