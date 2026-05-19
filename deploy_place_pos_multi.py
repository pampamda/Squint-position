"""
Deploy a PlacePosMultiShape policy on a real SO-101 arm.

The place policy starts from rest_qpos with the arm ALREADY holding the object
(end-state of a successful pick). A custom reset function (_place_hold_reset)
resets the sim episode without moving the real robot.

The last 6 dims of the state vector are injected each step:
    target_pos(3)    — provided via CLI --target_x/y/z
    tcp_to_target(3) — computed from live sim FK

Usage
-----
    python deploy_place_pos_multi.py \\
        --checkpoint runs/place_pos_multi__DATE/ckpt.pt \\
        --target_x 0.20 --target_y 0.10 --target_z 0.013

    # Stacking: supply target_pos above base object's top surface
    python deploy_place_pos_multi.py \\
        --checkpoint runs/place_pos_multi__DATE/ckpt.pt \\
        --target_x 0.25 --target_y 0.00 --target_z 0.055
"""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401 — registers custom gym environments
from deploy import create_wrist_camera_preprocessor, setup_safe_exit
from deploy_pos import inject_place_features
from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot
from train_squint import DeployAgent


@dataclass
class Args:
    checkpoint: Optional[str] = None
    env_id: str = "SO101PlacePosMultiShape-v1"
    obs_mode: str = "rgb+segmentation+state"
    control_mode: str = "pd_joint_target_delta_pos"
    max_episode_steps: int = 100
    control_freq: Optional[int] = 15
    action_scale: float = 0.15
    image_size: int = 128
    seed: int = 1
    episodes: int = 5

    # Target position (robot base frame, origin = base link centre).
    # Flat table placement: z ≈ item_half_height above table surface (~0.013 for cube).
    # Stacking: z = base_obj_pos.z + base_half_height + held_half_height.
    target_x: float = 0.20
    target_y: float = 0.10
    target_z: float = 0.013

    # W&B checkpoint loading
    wandb_entity: Optional[str] = None
    wandb_project: str = "maniskill-so101"
    wandb_agent_name: str = "squint_place_pos_multi"
    wandb_version: str = "latest"


def _place_hold_reset(env, seed=None, options=None):
    """Reset the place sim episode without moving the real robot.

    The real arm is already at rest_qpos holding the object (after pick()).
    Only the sim is reset; Sim2RealEnv then syncs sim joints from the real arm.
    """
    env.sim_env.reset(seed=seed, options=options)
    # Do NOT call env.agent.reset() — real robot must stay still.


def main(args: Args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    real_robot = create_real_robot()
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="sensors",
        max_episode_steps=args.max_episode_steps,
        domain_randomization=False,
        reward_mode="none",
        control_mode=args.control_mode,
        sensor_configs=dict(width=args.image_size, height=args.image_size),
    )
    sim_env = gym.make(args.env_id, **env_kwargs)
    sim_env = FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)

    preprocessor = create_wrist_camera_preprocessor(sim_env.unwrapped)
    real_env = Sim2RealEnv(
        sim_env=sim_env,
        agent=real_agent,
        control_freq=args.control_freq,
        sensor_data_preprocessing_function=preprocessor,
        real_reset_function=_place_hold_reset,
    )
    setup_safe_exit(sim_env, real_env, real_agent, recorder=None)

    real_obs, _ = real_env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DeployAgent(sim_env, sample_obs=real_obs).to(device)

    if args.checkpoint:
        checkpoint_config = {
            "wandb_entity": args.wandb_entity,
            "wandb_project_name": args.wandb_project,
            "agent_name": args.wandb_agent_name,
            "env_id": args.env_id,
            "seed": args.seed,
            "version": args.wandb_version,
        }
        agent.load_checkpoint(args.checkpoint, checkpoint_config)

    target_pos = torch.tensor(
        [[args.target_x, args.target_y, args.target_z]], dtype=torch.float32, device=device
    )

    for episode in range(args.episodes):
        real_obs, _ = real_env.reset()
        print(f"[Episode {episode}] target={target_pos.cpu().numpy().squeeze().tolist()}")
        for _ in tqdm(range(args.max_episode_steps), desc="Steps"):
            obs = {k: v.to(device) for k, v in real_obs.items()}

            tcp_pos = sim_env.unwrapped.agent.tcp_pos.to(device)
            tcp_to_target = target_pos - tcp_pos
            obs["state"] = inject_place_features(obs["state"], target_pos, tcp_to_target)

            action = agent.get_action(obs)
            scaled_action = np.clip(action.cpu().numpy() * args.action_scale, -1, 1)
            real_obs, _, terminated, truncated, _ = real_env.step(scaled_action)
            if bool(terminated) or bool(truncated):
                break

    for env in [sim_env, real_env]:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main(tyro.cli(Args))
