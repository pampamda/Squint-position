"""
Multi-shape position training for SO101 Lift.

Trains a single policy to grasp cube, sphere, capsule, and cylinder simultaneously.
Each parallel env is assigned one fixed shape; with 1024 envs, ~256 envs run each
shape every step, so gradients from all shapes update the same network weights.

Usage
-----
From scratch (recommended):
    python train_squint_multi.py

From scratch with domain randomization:
    python train_squint_multi.py --env-domain-randomization

Continue from an existing multi-shape checkpoint (same state dim):
    python train_squint_multi.py --checkpoint runs/lift_multi__20260517/ckpt.pt

Fine-tune from cube/LiftPos checkpoint (different state dim — partial load):
    # Step 1: brief from-scratch run to create an initial multi-shape ckpt
    # --eval_freq 10000 forces the ckpt to be written within 50k steps
    python train_squint_multi.py --total_timesteps 50000 --eval_freq 10000

    # Step 2: patch encoder + compatible weights from cube ckpt
    python patch_ckpt.py \\
        --src runs/lift_pos__20260507/ckpt.pt \\
        --dst runs/lift_multi__DATE/ckpt.pt

    # Step 3: resume full training
    python train_squint_multi.py --checkpoint runs/lift_multi__DATE/ckpt.pt
"""

import subprocess
import sys
from typing import Optional


def main():
    forward_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--finetune_from":
            # Intercept --finetune_from and print instructions instead of forwarding
            src = sys.argv[i + 1] if i + 1 < len(sys.argv) else "<cube_ckpt>"
            print(
                f"\n[train_squint_multi] --finetune_from={src}\n"
                "State dim changed (item_shape_id added). Use patch_ckpt.py instead:\n\n"
                "  # 1. brief from-scratch run (--eval_freq 10000 ensures ckpt is written)\n"
                "  python train_squint_multi.py --total_timesteps 50000 --eval_freq 10000\n\n"
                "  # 2. patch weights\n"
                f"  python patch_ckpt.py --src {src} --dst runs/lift_multi__DATE/ckpt.pt\n\n"
                "  # 3. resume full training\n"
                "  python train_squint_multi.py --checkpoint runs/lift_multi__DATE/ckpt.pt\n"
            )
            raise SystemExit(0)
        else:
            forward_args.append(sys.argv[i])
            i += 1

    default_args = [
        "--env_id", "SO101LiftMultiShape-v1",
        "--obs_mode", "rgb+segmentation+state",
        "--agent_name", "squint_multi",
        "--wandb_group", "SQUINT_MULTI",
        "--exp_name", "lift_multi",
        "--no-env-domain-randomization",
    ]
    cmd = [sys.executable, "train_squint.py", *default_args, *forward_args]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()