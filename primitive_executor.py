"""
High-level primitive executor for the squint robot system.

Wraps trained RL checkpoints into simple Python method calls.
Each call runs one full robot episode and returns True/False.

Callable primitives
-------------------
    pick(object_pos)      -- grasp and lift an object from the given 3D position
    place(target_pos)     -- carry held object to target_pos and release
    stack(target_pos)     -- place held object on top of another object at target_pos
                            (same policy as place; caller computes the stacking height)
    home()                -- return arm to rest_qpos via scripted joint interpolation
    detect_dummy(pos)     -- fixed-position stub for testing before real detection

Stage 2 usage
-------------
    executor = PrimitiveExecutor(
        pick_checkpoint="runs/lift_multi__DATE/ckpt.pt",
        place_checkpoint="runs/place_pos_multi__DATE/ckpt.pt",
    )
    obj_pos = executor.detect_dummy([0.30, 0.00, 0.013])
    ok = executor.pick(obj_pos)
    if ok:
        executor.place([0.20, 0.10, 0.013])          # flat table placement
        # or: executor.stack([0.25, 0.00, 0.040])    # on top of another object
        executor.home()
    executor.close()

Deploy conventions
------------------
    pick:  injects last 7 dims of state: object_pos_est(3) + tcp_to_obj(3) + valid(1)
    place: injects last 6 dims of state: target_pos(3) + tcp_to_target(3)
           is_item_grasped (the 7th-from-last dim) is NOT injected; comes from sim physics.
"""

import atexit
import signal
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401 — registers custom gym environments
from deploy import create_wrist_camera_preprocessor, silent_reset
from deploy_pos import inject_position_features, inject_place_features
from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot
from train_squint import DeployAgent


# ── Position providers ────────────────────────────────────────────────────────

class _PickPositionProvider:
    """Object position provider; updated once per pick() call."""

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


class _PlaceTargetProvider:
    """Target position provider; updated once per place() / stack() call."""

    def __init__(self):
        self._target = np.array([0.20, 0.00, 0.013], dtype=np.float32)

    def set(self, target_pos: np.ndarray):
        self._target = np.asarray(target_pos, dtype=np.float32).reshape(3)

    def get_features(
        self, sim_env, real_obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = real_obs["state"].device
        target = torch.tensor([self._target], device=device, dtype=torch.float32)
        tcp = sim_env.unwrapped.agent.tcp_pos.to(device).clone()
        tcp_to_target = target - tcp
        return target, tcp_to_target


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sim_env(
    env_id: str,
    obs_mode: str,
    control_mode: str,
    max_episode_steps: int,
    image_size: int,
) -> FlattenRGBDObservationWrapper:
    sim_env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="sensors",
        max_episode_steps=max_episode_steps,
        domain_randomization=False,
        reward_mode="none",
        control_mode=control_mode,
        sensor_configs=dict(width=image_size, height=image_size),
    )
    return FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)


def _place_hold_reset(env, seed=None, options=None):
    """Reset the place sim to a new episode without moving the real robot.

    Used as real_reset_function for the place Sim2RealEnv. The real robot stays at
    its current position (rest_qpos, gripper closed, holding item from pick).
    Sim2RealEnv.reset() then syncs the sim arm from the real robot's actual qpos.
    """
    env.sim_env.reset(seed=seed, options=options)
    # Do NOT call env.agent.reset() — real robot must not move between pick() and place().


# ── Executor ──────────────────────────────────────────────────────────────────

class PrimitiveExecutor:
    """
    High-level callable interface for pick / place / stack / home primitives.

    Loads trained checkpoints once at construction. Each primitive is a
    single blocking method call that drives the real robot through one
    episode and returns True (success) or False (timeout / failure).

    pick  uses SO101LiftMultiShape-v1    (object position injected each step).
    place uses SO101PlacePosMultiShape-v1 (target position injected each step).
    stack uses the same place checkpoint  (caller sets target height above base object).
    home  uses scripted joint interpolation (no policy or visual input required).

    Both pick_checkpoint and place_checkpoint are optional so that each
    primitive can be tested independently.
    """

    def __init__(
        self,
        pick_checkpoint: Optional[str] = None,
        place_checkpoint: Optional[str] = None,
        pick_env_id: str = "SO101LiftMultiShape-v1",
        place_env_id: str = "SO101PlacePosMultiShape-v1",
        obs_mode: str = "rgb+segmentation+state",
        control_mode: str = "pd_joint_target_delta_pos",
        max_episode_steps: int = 100,
        control_freq: int = 15,
        action_scale: float = 0.15,
        image_size: int = 128,
    ):
        if pick_checkpoint is None and place_checkpoint is None:
            raise ValueError(
                "At least one of pick_checkpoint or place_checkpoint must be provided."
            )

        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Real robot hardware (shared across all primitives) ────────
        self._real_robot = create_real_robot()
        self._real_robot.connect()
        self._real_agent = LeRobotRealAgent(self._real_robot)
        # Separate agent for place Sim2RealEnv: each Sim2RealEnv.__init__ sets
        # agent._sim_agent, so sharing one LeRobotRealAgent between two instances
        # would overwrite the first's _sim_agent reference.
        self._place_real_agent = LeRobotRealAgent(self._real_robot)

        # ── Pick primitive ────────────────────────────────────────────
        # SO101LiftMultiShape-v1: last 7 state dims injected each step.
        self._pick_sim_env = None
        self._pick_real_env = None
        self._pick_agent = None
        if pick_checkpoint is not None:
            self._pick_sim_env = _make_sim_env(
                pick_env_id, obs_mode, control_mode, max_episode_steps, image_size
            )
            preprocessor = create_wrist_camera_preprocessor(self._pick_sim_env.unwrapped)
            self._pick_real_env = Sim2RealEnv(
                sim_env=self._pick_sim_env,
                agent=self._real_agent,
                control_freq=control_freq,
                sensor_data_preprocessing_function=preprocessor,
                real_reset_function=silent_reset,
            )
            sample_obs, _ = self._pick_real_env.reset()
            self._pick_agent = DeployAgent(
                self._pick_sim_env, sample_obs=sample_obs
            ).to(self.device)
            self._pick_agent.load_checkpoint(pick_checkpoint)
            print(f"[PrimitiveExecutor] pick  loaded: {pick_checkpoint}")

        # ── Place / Stack primitive ───────────────────────────────────
        # SO101PlacePosMultiShape-v1: last 6 state dims injected each step.
        # stack() reuses this same agent with a higher target_pos.
        self._place_sim_env = None
        self._place_real_env = None
        self._place_agent = None
        if place_checkpoint is not None:
            self._place_sim_env = _make_sim_env(
                place_env_id, obs_mode, control_mode, max_episode_steps, image_size
            )
            preprocessor = create_wrist_camera_preprocessor(self._place_sim_env.unwrapped)
            self._place_real_env = Sim2RealEnv(
                sim_env=self._place_sim_env,
                agent=self._place_real_agent,
                control_freq=control_freq,
                sensor_data_preprocessing_function=preprocessor,
                real_reset_function=_place_hold_reset,
            )
            sample_obs, _ = self._place_real_env.reset()
            self._place_agent = DeployAgent(
                self._place_sim_env, sample_obs=sample_obs
            ).to(self.device)
            self._place_agent.load_checkpoint(place_checkpoint)
            print(f"[PrimitiveExecutor] place loaded: {place_checkpoint}")

        # ── Per-primitive position providers ──────────────────────────
        self._pick_pos_provider = _PickPositionProvider()
        self._place_target_provider = _PlaceTargetProvider()

        # ── Graceful shutdown ─────────────────────────────────────────
        atexit.register(self.close)
        signal.signal(signal.SIGINT, lambda sig, frame: (self.close(), sys.exit(0)))

    # ── Detector interface ────────────────────────────────────────────────────

    def detect_dummy(self, pos_xyz: list = [0.30, 0.00, 0.013]) -> np.ndarray:
        """Return a manually specified fixed position (no real detection).

        Use during initial testing before the RGB-D detector is integrated.
        Coordinate frame: robot base frame (origin = base link centre).
        Typical training range: X in [0.20, 0.40], Y in [-0.10, 0.10], Z ≈ 0.013 m.

        Args:
            pos_xyz: [x, y, z] in metres, robot base frame.
        Returns:
            (3,) float32 ndarray — passed straight through to pick().
        """
        return np.array(pos_xyz, dtype=np.float32)

    # ── Primitives ────────────────────────────────────────────────────────────

    def pick(self, object_pos: np.ndarray) -> bool:
        """Grasp and lift the object at object_pos, ending at rest_qpos.

        Uses SO101LiftMultiShape-v1. The object position is injected into the
        state observation each step (7 dims: object_pos_est(3) + tcp_to_obj(3)
        + position_valid(1)). The policy conditions on item_shape_id and adapts
        its grasp strategy per geometry from visual features.

        The arm ends at rest_qpos holding the object. Call place() or stack() next.

        Args:
            object_pos: (3,) [x, y, z] in robot base frame (metres).
                        Obtain from detect_dummy() or a real RGB-D detector.
        Returns:
            True  — item grasped and arm at rest_qpos.
            False — episode timed out or grasp failed.
        """
        if self._pick_agent is None:
            raise RuntimeError("pick_checkpoint was not provided at construction.")

        self._pick_pos_provider.set(object_pos)
        real_obs, _ = self._pick_real_env.reset()

        for _ in range(self.max_episode_steps):
            obs = {k: v.to(self.device) for k, v in real_obs.items()}
            obj_pos_t, tcp_to_obj, valid = self._pick_pos_provider.get_position(
                self._pick_sim_env, obs
            )
            obs["state"] = inject_position_features(obs["state"], obj_pos_t, tcp_to_obj, valid)
            action = self._pick_agent.get_action(obs)
            scaled = np.clip(action.cpu().numpy() * self.action_scale, -1, 1)
            real_obs, _, terminated, truncated, _ = self._pick_real_env.step(scaled)
            if terminated:
                return True
            if truncated:
                return False

        return False

    def place(self, target_pos: np.ndarray) -> bool:
        """Carry the held object to target_pos and release it there.

        Uses SO101PlacePosMultiShape-v1. The target position and tcp_to_target
        vector are injected into the last 6 dims of the state observation each
        step; is_item_grasped (the preceding dim) comes from sim physics.

        target_pos should be on the table surface:
            [x, y, table_height + item_half_height]

        The arm must be at rest_qpos holding an object (terminal state of pick()).

        Args:
            target_pos: (3,) [x, y, z] in robot base frame (metres).
        Returns:
            True  — object released within 3 cm of target_pos.
            False — episode timed out or placement failed.
        """
        if self._place_agent is None:
            raise RuntimeError("place_checkpoint was not provided at construction.")

        self._place_target_provider.set(target_pos)
        real_obs, _ = self._place_real_env.reset()

        for _ in range(self.max_episode_steps):
            obs = {k: v.to(self.device) for k, v in real_obs.items()}
            target_t, tcp_to_target = self._place_target_provider.get_features(
                self._place_sim_env, obs
            )
            obs["state"] = inject_place_features(obs["state"], target_t, tcp_to_target)
            action = self._place_agent.get_action(obs)
            scaled = np.clip(action.cpu().numpy() * self.action_scale, -1, 1)
            real_obs, _, terminated, truncated, _ = self._place_real_env.step(scaled)
            if terminated:
                return True
            if truncated:
                return False

        return False

    def stack(self, target_pos: np.ndarray) -> bool:
        """Place the held object on top of another object at target_pos.

        Uses the same SO101PlacePosMultiShape-v1 policy as place().
        The difference is that target_pos should be above the base object's
        top surface, not on the table:

            target_pos = base_obj_pos + [0, 0, base_half_height + held_half_height]

        The upstream planner is responsible for computing this height.
        If stacking accuracy is insufficient, a dedicated StackPos policy
        can replace this method without changing the caller interface.

        Args:
            target_pos: (3,) [x, y, z] in robot base frame (metres).
        Returns:
            True  — held object released within 3 cm of target_pos.
            False — episode timed out or stacking failed.
        """
        return self.place(target_pos)

    def home(self) -> bool:
        """Return arm to rest_qpos via scripted joint interpolation.

        Interpolates at 30 Hz with max 0.025 rad/step until the arm reaches
        rest_qpos or the 20-second budget is exhausted. No policy or visual
        input is required.

        Must be called after every place() or stack() so the arm is at
        rest_qpos when the next pick() begins, matching the lift policy's
        initial-state distribution.

        Returns:
            True  — arm reached rest_qpos.
            False — interpolation call failed.
        """
        sim_env = self._pick_sim_env or self._place_sim_env
        # rest_qpos is the start keyframe (open gripper) — the ready-to-pick position.
        # keyframes["rest"] is the closed-gripper rest, which is NOT what we want here.
        rest_qpos = sim_env.unwrapped.rest_qpos.cpu().numpy()
        try:
            self._real_agent.reset(rest_qpos)
            return True
        except Exception as e:
            print(f"[PrimitiveExecutor] home() failed: {e}")
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Return robot to rest and release all resources."""
        try:
            self.home()
        except Exception:
            pass
        for env in [
            self._pick_real_env,
            self._place_real_env,
            self._pick_sim_env,
            self._place_sim_env,
        ]:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
        try:
            self._real_robot.disconnect()
        except Exception:
            pass


# ── Quick test script ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test pick → place/stack → home with dummy detector"
    )
    parser.add_argument("--pick_checkpoint", default=None, help="Path to lift_multi ckpt.pt")
    parser.add_argument("--place_checkpoint", default=None, help="Path to place_pos_multi ckpt.pt")
    parser.add_argument("--obj_x", type=float, default=0.30)
    parser.add_argument("--obj_y", type=float, default=0.00)
    parser.add_argument("--obj_z", type=float, default=0.013)
    parser.add_argument("--target_x", type=float, default=0.20)
    parser.add_argument("--target_y", type=float, default=0.10)
    parser.add_argument("--target_z", type=float, default=0.013)
    parser.add_argument("--mode", choices=["place", "stack"], default="place",
                        help="place: flat table; stack: elevated target height")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--action_scale", type=float, default=0.15)
    parser.add_argument("--control_freq", type=int, default=15)
    args = parser.parse_args()

    executor = PrimitiveExecutor(
        pick_checkpoint=args.pick_checkpoint,
        place_checkpoint=args.place_checkpoint,
        action_scale=args.action_scale,
        control_freq=args.control_freq,
    )

    obj_pos = np.array([args.obj_x, args.obj_y, args.obj_z], dtype=np.float32)
    target_pos = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)

    results = {"pick": [], "place": [], "stack": []}
    for ep in range(args.episodes):
        print(f"\n[Episode {ep + 1}/{args.episodes}]")

        ok_pick = True
        if executor._pick_agent is not None:
            pos = executor.detect_dummy(obj_pos.tolist())
            print(f"  pick  at {pos}")
            ok_pick = executor.pick(pos)
            results["pick"].append(ok_pick)
            print(f"  pick  → {'SUCCESS' if ok_pick else 'FAILED'}")

        if ok_pick and executor._place_agent is not None:
            print(f"  {args.mode} at {target_pos}")
            if args.mode == "stack":
                ok = executor.stack(target_pos)
                results["stack"].append(ok)
            else:
                ok = executor.place(target_pos)
                results["place"].append(ok)
            print(f"  {args.mode} → {'SUCCESS' if ok else 'FAILED'}")

        executor.home()
        print("  home  → done")

    if results["pick"]:
        print(f"\nPick:  {sum(results['pick'])}/{len(results['pick'])} succeeded")
    if results["place"]:
        print(f"Place: {sum(results['place'])}/{len(results['place'])} succeeded")
    if results["stack"]:
        print(f"Stack: {sum(results['stack'])}/{len(results['stack'])} succeeded")

    executor.close()
