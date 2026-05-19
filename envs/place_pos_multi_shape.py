from dataclasses import asdict, dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .lift import Lift
from .lift_multi_shape import LiftMultiShape, LiftMultiShapeRandomizationConfig, NUM_SHAPES

# SO101 gripper joint value for a closed grip (matches 'rest' keyframe in so101.py).
# Servo calibration: Sim -10° <-> Servo closed. Using this as episode-start position
# ensures the PD controller squeezes the spawned item, giving is_item_grasped=True.
GRIPPER_HOLD_QPOS = -10 * np.pi / 180


@dataclass
class PlacePosMultiShapeRandomizationConfig(LiftMultiShapeRandomizationConfig):
    """Adds grasp-offset noise to simulate holding-start distribution."""
    grasp_offset_noise_std: Sequence[float] = (0.005, 0.005, 0.003)
    grasp_offset_clip: float = 0.015


class PlacePosMultiShape(LiftMultiShape):
    """
    Multi-shape position-based place environment.

    Episode start: arm at rest_qpos, object spawned at TCP position (simulating a held object).
    Episode goal: carry the object to target_pos and release it there.

    target_pos is uniformly sampled on the table during training.
    At deploy time, the last 6 dims of the state vector are replaced:
        target_pos(3) + tcp_to_target(3)
    is_item_grasped (the preceding 1 dim) is left as-is from sim physics.

    Both place(target_pos) and stack(target_pos) in PrimitiveExecutor use
    this policy. Stack passes a target_pos above another object's surface;
    no separate training environment is needed.
    """

    def __init__(
        self,
        *args,
        spawn_box_pos=(0.3, 0),
        spawn_box_half_size=0.1,
        domain_randomization_config: Union[
            PlacePosMultiShapeRandomizationConfig, dict
        ] = PlacePosMultiShapeRandomizationConfig(),
        **kwargs,
    ):
        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size
        if isinstance(domain_randomization_config, dict):
            merged = asdict(PlacePosMultiShapeRandomizationConfig())
            common.dict_merge(merged, domain_randomization_config)
            domain_randomization_config = dacite.from_dict(
                data_class=PlacePosMultiShapeRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        super().__init__(
            *args,
            domain_randomization_config=domain_randomization_config,
            **kwargs,
        )

    # ── Scene loading ─────────────────────────────────────────────────────────

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # Persistent buffer for target positions, updated each episode
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Orange goal marker (hidden from sensor, shown in render)
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.015,
            material=sapien.render.RenderMaterial(base_color=[1.0, 0.5, 0.0, 1.0]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    # ── Episode initialisation ────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # PlacePosMultiShape → LiftMultiShape → Lift → DefaultCameraEnv.
        # We want LiftMultiShape._load_scene (actors, shape assignment, half_sizes)
        # but NOT Lift._initialize_episode (which places the item on the table).
        # super(Lift, self) resolves one level above Lift in the MRO, landing on
        # DefaultCameraEnv._initialize_episode. If the inheritance chain changes,
        # update this call accordingly.
        super(Lift, self)._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Arm at rest_qpos; gripper CLOSED so the PD controller squeezes the
            # item spawned below, giving is_item_grasped=True from the first step.
            # rest_qpos[-1] is the start-keyframe value (60° = open) — override it.
            init_qpos = self.rest_qpos.clone()
            init_qpos[-1] = GRIPPER_HOLD_QPOS
            self.agent.robot.set_qpos(
                init_qpos
                + torch.randn((b, init_qpos.shape[-1]), device=self.device)
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0.0], device=self.device
            )
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            # Sample target_pos on the table, clear of the item drop zone
            target_clearance = self.item_half_sizes.max().item() + 0.03
            target_xy_offset = sampler.sample(target_clearance, 100)

            target_xyz = torch.zeros((b, 3), device=self.device)
            target_xyz[:, :2] = spawn_center[env_idx, :2] + target_xy_offset
            target_xyz[:, 2] = self.item_half_sizes[env_idx]  # flat on table
            self.target_pos[env_idx] = target_xyz
            self.goal_site.set_pose(Pose.create_from_pq(target_xyz))

            # Spawn item at TCP position to simulate a held object.
            # grasp_offset_noise covers the real distribution of pick outcomes.
            cfg: PlacePosMultiShapeRandomizationConfig = self.domain_randomization_config
            tcp_pos = self.agent.tcp_pos[env_idx]  # FK is updated immediately after set_qpos
            std = torch.tensor(cfg.grasp_offset_noise_std, device=self.device)
            noise = torch.clamp(
                torch.randn((b, 3), device=self.device) * std,
                -cfg.grasp_offset_clip,
                cfg.grasp_offset_clip,
            )
            item_pos = tcp_pos + noise
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_pos, qs))

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict):
        obs = {}
        if self.obs_mode_struct.state:
            b = self.num_envs

            # DR fields first (same ordering convention as lift_multi_shape)
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )

            # Base kinematic state + shape identity
            shape_one_hot = torch.zeros((b, NUM_SHAPES), device=self.device)
            shape_one_hot.scatter_(1, self.item_shape_ids.unsqueeze(1), 1.0)
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pos=self.agent.tcp_pose.raw_pose,
                dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1] - self.rest_qpos[:-1],
                item_shape_id=shape_one_hot,
            )

            # Grasp state: from physics, NOT injected at deploy
            obs["is_item_grasped"] = info["is_item_grasped"].float().unsqueeze(-1)

            # LAST 6 dims — injected at deploy time by inject_place_features():
            #   target_pos(3) + tcp_to_target(3)
            target = self.target_pos  # (B, 3)
            obs.update(
                target_pos=target,
                tcp_to_target=target - self.agent.tcp_pos,
            )
        return obs

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self):
        item_pos = self.item.pose.p
        dist = torch.linalg.norm(item_pos - self.target_pos, dim=1)

        is_at_target = dist < 0.03
        is_item_grasped = self.agent.is_grasping(self.item)
        is_robot_static = self.agent.is_static()
        item_lifted = item_pos[:, 2] >= (self.item_half_sizes + 1e-3)
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_item = self.agent.is_touching(self.item)

        # is_item_grasped=False: gripper has released the item (grip force gone).
        # ~robot_touching_item: no residual contact (e.g. arm nudging item after release).
        # Both conditions guard against counting release transients as success.
        success = is_at_target & (~is_item_grasped) & (~robot_touching_item) & is_robot_static

        return {
            "success": success,
            "is_at_target": is_at_target,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "item_to_target_dist": dist,
            "item_lifted": item_lifted,
            "robot_touching_table": robot_touching_table,
            "robot_touching_item": robot_touching_item,
        }

    # ── Reward ────────────────────────────────────────────────────────────────

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        item_pos = self.item.pose.p
        target = self.target_pos

        # 3-D distance to target
        item_to_target_dist = torch.linalg.norm(target - item_pos, dim=1)
        place_reward = 1 - torch.tanh(5.0 * item_to_target_dist)

        # Staged Z approach: go above target first, then lower
        item_to_target_xy = torch.linalg.norm(target[:, :2] - item_pos[:, :2], dim=1)
        dist_z_far = torch.abs((target[:, 2] + 0.05) - item_pos[:, 2])
        dist_z_close = torch.abs(target[:, 2] - item_pos[:, 2])
        dist_z = torch.where(item_to_target_xy < 0.05, dist_z_close, dist_z_far)
        place_reward = place_reward + (1 - torch.tanh(10.0 * dist_z))

        # Gripper openness (encourages release when at target)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (
            (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)
        )

        # Robot static reward (stabilises after release)
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], dim=1)
        static_reward = 1 - torch.tanh(robot_v * 10)

        reward = torch.zeros(item_pos.shape[0], device=self.device)

        # Phase 1 — holding and moving toward target
        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        # Phase 2 — near target: encourage releasing
        item_near_target = item_to_target_dist < 0.05
        is_dropped = (~info["robot_touching_item"]).float()
        reward[item_near_target] = (
            4 + place_reward + is_dropped + gripper_openness + static_reward
        )[item_near_target]

        # Success
        reward[info["success"]] = 9

        # Penalty for table contact
        reward -= 6 * info["robot_touching_table"].float()

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101PlacePosMultiShape-v1", max_episode_steps=100)
class PlacePosMultiShapeEnv(PlacePosMultiShape):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
