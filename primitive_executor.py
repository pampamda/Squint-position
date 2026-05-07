"""
High-level primitive executor for the squint robot system.

Wraps trained RL checkpoints into simple Python method calls.
Each call runs one full robot episode and returns True/False.

Quick test with dummy (fixed) position:
    executor = PrimitiveExecutor(
        pick_checkpoint="runs/lift_pos__20260430/ckpt.pt"
    )
    pos = executor.detect_dummy([0.30, 0.00, 0.013])
    success = executor.pick(pos)
    executor.close()
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Optional

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401 — registers custom gym environments
from deploy import create_wrist_camera_preprocessor, setup_safe_exit, silent_reset
from deploy_pos import inject_position_features
from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot
from train_squint import DeployAgent


class _RuntimePositionProvider:
    """Position provider whose target position can be set at each call."""

    def __init__(self):
        self._pos = np.array([0.30, 0.00, 0.013], dtype=np.float32)

    def set(self, pos: np.ndarray):
        self._pos = np.asarray(pos, dtype=np.float32).reshape(3)

    def get_position(
        self, sim_env, real_obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = real_obs["state"].device
        obj = torch.tensor([self._pos], device=device, dtype=torch.float32)
        tcp = sim_env.unwrapped.agent.tcp_pos.to(device).clone()
        tcp_to_obj = obj - tcp
        valid = torch.ones((1, 1), device=device)
        return obj, tcp_to_obj, valid


class PrimitiveExecutor:
    """
    High-level callable interface for pick / (future: place, stack) primitives.

    Loads trained checkpoints once at __init__. After that, each primitive
    is a single blocking method call that drives the real robot through one
    full episode and returns success (True) or failure (False).

    Example
    -------
    executor = PrimitiveExecutor(
        pick_checkpoint="runs/lift_pos__20260430/ckpt.pt"
    )
    # --- phase 1: dummy detector (fixed position) ---
    pos = executor.detect_dummy([0.30, 0.00, 0.013])
    ok  = executor.pick(pos)
    print("pick", "OK" if ok else "FAILED")

    # --- future: swap in real RGB-D detector ---
    # pos = real_detector.get_object_pos()
    # ok  = executor.pick(pos)

    executor.close()
    """

    def __init__(
        self,
        pick_checkpoint: str,
        env_id: str = "SO101LiftCubePos-v1",
        obs_mode: str = "rgb+segmentation+state",
        control_mode: str = "pd_joint_target_delta_pos",
        max_episode_steps: int = 120,
        control_freq: int = 15,
        action_scale: float = 0.15,
        image_size: int = 128,
    ):
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Real robot ────────────────────────────────────────────────
        self._real_robot = create_real_robot()
        self._real_robot.connect()
        self._real_agent = LeRobotRealAgent(self._real_robot)

        # ── Simulation env (state shape + wrist camera config) ────────
        env_kwargs = dict(
            obs_mode=obs_mode,
            render_mode="sensors",
            max_episode_steps=max_episode_steps,
            domain_randomization=False,
            reward_mode="none",
            control_mode=control_mode,
            sensor_configs=dict(width=image_size, height=image_size),
        )
        self._sim_env = gym.make(env_id, **env_kwargs)
        self._sim_env = FlattenRGBDObservationWrapper(
            self._sim_env, rgb=True, depth=False, state=True
        )

        preprocessor = create_wrist_camera_preprocessor(self._sim_env.unwrapped)
        self._real_env = Sim2RealEnv(
            sim_env=self._sim_env,
            agent=self._real_agent,
            control_freq=control_freq,
            sensor_data_preprocessing_function=preprocessor,
            real_reset_function=silent_reset,
        )
        setup_safe_exit(self._sim_env, self._real_env, self._real_agent)

        # ── Pick agent ────────────────────────────────────────────────
        sample_obs, _ = self._real_env.reset()
        self._pick_agent = DeployAgent(
            self._sim_env, sample_obs=sample_obs
        ).to(self.device)
        self._pick_agent.load_checkpoint(pick_checkpoint)

        # ── Shared runtime position provider ─────────────────────────
        self._pos_provider = _RuntimePositionProvider()

        print(f"[PrimitiveExecutor] Ready. pick_checkpoint={pick_checkpoint}")

    # ──────────────────────────────────────────────────────────────────
    # Detector interface
    # ──────────────────────────────────────────────────────────────────

    def detect_dummy(self, pos_xyz: list = [0.30, 0.00, 0.013]) -> np.ndarray:
        """Return a manually specified fixed position (no real detection).

        Use this during initial testing before implementing real RGB-D detection.

        Coordinate frame: robot base frame (origin = base link centre).
        Typical training range: X∈[0.20, 0.40], Y∈[-0.10, 0.10], Z≈0.013 m.

        Args:
            pos_xyz: [x, y, z] in metres, robot base frame.

        Returns:
            (3,) float32 array — the position passed straight through.
        """
        return np.array(pos_xyz, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Primitives
    # ──────────────────────────────────────────────────────────────────

    def pick(self, object_pos: np.ndarray) -> bool:
        """Execute the pick primitive: move to object_pos, grasp, lift to rest.

        Args:
            object_pos: (3,) array [x, y, z] in robot base frame.
                        Use detect_dummy() or a real detector to obtain this.

        Returns:
            True  — episode terminated with success (sim reports item grasped
                    and robot at rest pose).
            False — episode timed out or failed.
        """
        self._pos_provider.set(object_pos)
        real_obs, _ = self._real_env.reset()

        for _ in range(self.max_episode_steps):
            obs = {k: v.to(self.device) for k, v in real_obs.items()}

            obj_pos, tcp_to_obj, valid = self._pos_provider.get_position(
                self._sim_env, obs
            )
            obs["state"] = inject_position_features(
                obs["state"], obj_pos, tcp_to_obj, valid
            )

            action = self._pick_agent.get_action(obs)
            scaled = np.clip(action.cpu().numpy() * self.action_scale, -1, 1)
            real_obs, _, terminated, truncated, _ = self._real_env.step(scaled)

            if terminated:
                return True
            if truncated:
                return False

        return False

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def close(self):
        """Return robot to rest and release all resources."""
        try:
            self._real_agent.reset(
                self._sim_env.unwrapped.agent.keyframes["rest"].qpos
            )
        except Exception:
            pass
        for env in [self._sim_env, self._real_env]:
            try:
                env.close()
            except Exception:
                pass
        try:
            self._real_robot.disconnect()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# Quick test script
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test pick primitive with dummy detector")
    parser.add_argument("--checkpoint", required=True, help="Path to lift_pos ckpt.pt")
    parser.add_argument("--x", type=float, default=0.30, help="Object X (m)")
    parser.add_argument("--y", type=float, default=0.00, help="Object Y (m)")
    parser.add_argument("--z", type=float, default=0.013, help="Object Z (m)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--action_scale", type=float, default=0.15)
    parser.add_argument("--control_freq", type=int, default=15)
    args = parser.parse_args()

    executor = PrimitiveExecutor(
        pick_checkpoint=args.checkpoint,
        action_scale=args.action_scale,
        control_freq=args.control_freq,
    )

    results = []
    for ep in range(args.episodes):
        pos = executor.detect_dummy([args.x, args.y, args.z])
        print(f"\n[Episode {ep+1}/{args.episodes}] pick at {pos}")
        ok = executor.pick(pos)
        results.append(ok)
        print(f"  → {'SUCCESS' if ok else 'FAILED'}")

    print(f"\nResults: {sum(results)}/{len(results)} succeeded")
    executor.close()
