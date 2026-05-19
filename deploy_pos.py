"""
Deploy a position-centric Lift policy with pluggable position providers.

MVP defaults:
- SO101LiftCubePos-v1
- SimGTPositionProvider (sim pose + noise)
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

from deploy import create_wrist_camera_preprocessor, setup_safe_exit, silent_reset
from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.position_provider import (
    FixedPositionProvider,
    RealDetectorPositionProvider,
    SimGTPositionProvider,
)
from deploy_utils.robot_config import create_real_robot
from train_squint import DeployAgent


@dataclass
class Args:
    checkpoint: Optional[str] = None
    env_id: str = "SO101LiftCubePos-v1"
    obs_mode: str = "rgb+segmentation+state"
    control_mode: str = "pd_joint_target_delta_pos"
    max_episode_steps: int = 120
    control_freq: Optional[int] = 30
    action_scale: float = 0.15
    image_size: int = 128
    seed: int = 1
    episodes: int = 5
    provider: str = "sim_gt"
    retry_steps: int = 3

    # sim-gt provider noise
    pos_noise_x: float = 0.005
    pos_noise_y: float = 0.005
    pos_noise_z: float = 0.003
    pos_noise_clip: float = 0.02
    pos_dropout_prob: float = 0.0

    # fixed provider: 手动指定物体坐标（世界坐标系，原点=机器人基座）
    # 训练时典型范围：X∈[0.2,0.4], Y∈[-0.1,0.1], Z≈0.013
    fixed_pos_x: float = 0.30
    fixed_pos_y: float = 0.00
    fixed_pos_z: float = 0.013

    # checkpoint loading from wandb
    wandb_entity: Optional[str] = None
    wandb_project: str = "maniskill-so101"
    wandb_agent_name: str = "squint_pos"
    wandb_version: str = "latest"


def inject_place_features(
    state: torch.Tensor,
    target_pos: torch.Tensor,
    tcp_to_target: torch.Tensor,
) -> torch.Tensor:
    """Replace final 6 dims with [target_pos(3), tcp_to_target(3)].

    Deploy convention: PlacePosMultiShape._get_obs_extra() places target_pos
    and tcp_to_target as the LAST 6 dims of the state vector.
    is_item_grasped (the dim immediately before these 6) is NOT injected —
    it comes from sim physics which tracks the real gripper state via FK.
    """
    fused = torch.cat([target_pos, tcp_to_target], dim=-1)
    if fused.shape[-1] != 6:
        raise ValueError(f"Place feature dim {fused.shape[-1]} != 6")
    if state.shape[-1] < 6:
        raise ValueError(f"State dim {state.shape[-1]} too small for place feature injection")
    out = state.clone()
    out[..., -6:] = fused
    return out


def inject_position_features(
    state: torch.Tensor,
    object_pos_est: torch.Tensor,
    tcp_to_object_pos: torch.Tensor,
    position_valid: torch.Tensor,
) -> torch.Tensor:
    """Replace final 7 dims with [obj_pos_est(3), tcp_to_obj(3), valid(1)].

    依赖约定：训练环境 LiftPosition._get_obs_extra() 的 state 末尾 7 维
    必须是 [object_pos_est(3), tcp_to_object_pos(3), position_valid(1)]。
    如果训练环境的 obs 结构发生变化，此函数需同步更新。
    """
    fused = torch.cat([object_pos_est, tcp_to_object_pos, position_valid], dim=-1)
    expected_pos_dim = 7  # object_pos_est(3) + tcp_to_object_pos(3) + position_valid(1)
    if fused.shape[-1] != expected_pos_dim:
        raise ValueError(f"Position feature dim {fused.shape[-1]} != expected {expected_pos_dim}")
    if state.shape[-1] < expected_pos_dim:
        raise ValueError(
            f"State dim {state.shape[-1]} smaller than position feature dim {expected_pos_dim}"
        )
    out = state.clone()
    out[..., -expected_pos_dim:] = fused
    return out


def build_provider(args: Args):
    if args.provider == "sim_gt":
        return SimGTPositionProvider(
            noise_std_xyz=(args.pos_noise_x, args.pos_noise_y, args.pos_noise_z),
            noise_clip=args.pos_noise_clip,
            dropout_prob=args.pos_dropout_prob,
        )
    if args.provider == "fixed":
        return FixedPositionProvider(
            object_pos_x=args.fixed_pos_x,
            object_pos_y=args.fixed_pos_y,
            object_pos_z=args.fixed_pos_z,
        )
    if args.provider == "real_detector":
        return RealDetectorPositionProvider()
    raise ValueError(f"Unknown provider: {args.provider}")


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
        real_reset_function=silent_reset,
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

    provider = build_provider(args)
    for episode in range(args.episodes):
        real_obs, _ = real_env.reset()
        print(f"[Episode {episode}]")
        for _ in tqdm(range(args.max_episode_steps), desc="Steps"):
            obs = {k: v.to(device) for k, v in real_obs.items()}
            obj_pos, tcp_to_obj, valid = provider.get_position(sim_env, obs)

            if (valid < 0.5).any():
                # Detection failure handling: small bounded retries, then no-op.
                for _retry in range(args.retry_steps):
                    obj_pos, tcp_to_obj, valid = provider.get_position(sim_env, obs)
                    if (valid >= 0.5).all():
                        break
                if (valid < 0.5).any():
                    action = torch.zeros_like(agent.get_action(obs))
                    real_obs, *_ = real_env.step(action.cpu().numpy())
                    continue

            obs["state"] = inject_position_features(obs["state"], obj_pos, tcp_to_obj, valid)
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

