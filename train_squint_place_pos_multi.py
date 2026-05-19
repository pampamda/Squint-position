"""
Multi-shape position-based place training for SO101.

Trains a single policy to carry and release cube, sphere, capsule, and cylinder
at arbitrary target coordinates. Each parallel env is assigned one fixed shape.

This checkpoint is used by both place(target_pos) and stack(target_pos) in
PrimitiveExecutor — stack passes a target_pos above another object's surface.

Usage
-----
From scratch:
    python train_squint_place_pos_multi.py

With domain randomization:
    python train_squint_place_pos_multi.py --env-domain-randomization

Reduce envs if GPU OOM (7-8 GB GPU):
    python train_squint_place_pos_multi.py --num_envs 512

Continue from checkpoint:
    python train_squint_place_pos_multi.py --checkpoint runs/place_pos_multi__DATE/ckpt.pt
"""

import subprocess
import sys


def main():
    default_args = [
        "--env_id", "SO101PlacePosMultiShape-v1",
        "--obs_mode", "rgb+segmentation+state",
        "--agent_name", "squint_place_pos_multi",
        "--wandb_group", "SQUINT_PLACE_POS_MULTI",
        "--exp_name", "place_pos_multi",
        "--no-env-domain-randomization",
    ]
    cmd = [sys.executable, "train_squint.py", *default_args, *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
