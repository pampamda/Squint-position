"""
Position-centric training entrypoint for SO101 Lift.

This wrapper reuses train_squint.py and injects MVP defaults:
- env_id=SO101LiftCubePos-v1
- obs_mode=rgb+segmentation+state
"""

import subprocess
import sys


def main():
    default_args = [
        "--env_id",
        "SO101LiftCubePos-v1",
        "--obs_mode",
        "rgb+segmentation+state",
        "--agent_name",
        "squint_pos",
        "--wandb_group",
        "SQUINT_POS",
        "--exp_name",
        "lift_pos",
        "--no-env-domain-randomization",
    ]
    cmd = [sys.executable, "train_squint.py", *default_args, *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

