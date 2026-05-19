"""
Multi-shape place training for SO101.

Trains a single policy to pick up and place cube, sphere, capsule, and cylinder
into a bin simultaneously. Each parallel env is assigned one fixed shape.

Usage
-----
From scratch:
    python train_squint_place_multi.py

With domain randomization:
    python train_squint_place_multi.py --env-domain-randomization

Reduce envs if GPU OOM (7-8 GB GPU):
    python train_squint_place_multi.py --num_envs 512

Continue from checkpoint:
    python train_squint_place_multi.py --checkpoint runs/place_multi__DATE/ckpt.pt
"""

import subprocess
import sys


def main():
    default_args = [
        "--env_id", "SO101PlaceMultiShape-v1",
        "--obs_mode", "rgb+segmentation+state",
        "--agent_name", "squint_place_multi",
        "--wandb_group", "SQUINT_PLACE_MULTI",
        "--exp_name", "place_multi",
        "--no-env-domain-randomization",
    ]
    cmd = [sys.executable, "train_squint.py", *default_args, *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
