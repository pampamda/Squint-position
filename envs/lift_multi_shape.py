from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .lift import Lift, LiftRandomizationConfig
from .lift_position import LiftPositionRandomizationConfig

# Shape constants
SHAPE_CUBE = 0
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_CYLINDER = 3
NUM_SHAPES = 4
SHAPE_NAMES = ["cube", "sphere", "capsule", "cylinder"]


@dataclass
class LiftMultiShapeRandomizationConfig(LiftPositionRandomizationConfig):
    """Extends position config with sphere and capsule geometry ranges."""
    sphere_radius_range: Sequence[float] = (0.018, 0.030)
    capsule_radius_range: Sequence[float] = (0.012, 0.020)
    capsule_half_length_range: Sequence[float] = (0.020, 0.040)


class LiftMultiShape(Lift):
    """
    Lift variant that simultaneously trains on cube, sphere, capsule, and cylinder.

    Each parallel env is assigned one fixed shape at scene-load time.
    With N parallel envs, ~N/4 envs run each shape every training step,
    so the shared policy learns all shapes simultaneously.

    Observations include item_shape_id (4-dim one-hot) so the policy can
    condition its grasp strategy on the object type.

    Position observations follow the same deploy convention as LiftPosition:
    object_pos_est / tcp_to_object_pos / position_valid are always last in state.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[
            LiftMultiShapeRandomizationConfig, dict
        ] = LiftMultiShapeRandomizationConfig(),
        **kwargs,
    ):
        # Handle dict config before super() so Lift's isinstance branch picks it up correctly
        if isinstance(domain_randomization_config, dict):
            merged = asdict(LiftMultiShapeRandomizationConfig())
            common.dict_merge(merged, domain_randomization_config)
            domain_randomization_config = dacite.from_dict(
                data_class=LiftMultiShapeRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        # item_type="mixed" bypasses Lift._load_scene()'s cube/can guard
        # because we override _load_scene() entirely
        super().__init__(
            *args,
            item_type="mixed",
            domain_randomization_config=domain_randomization_config,
            **kwargs,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _position_noise_tensor(self, b: int) -> torch.Tensor:
        cfg: LiftMultiShapeRandomizationConfig = self.domain_randomization_config
        std = torch.tensor(cfg.position_noise_std, device=self.device, dtype=torch.float32)
        noise = torch.randn((b, 3), device=self.device) * std
        if cfg.position_noise_clip > 0:
            noise = torch.clamp(noise, -cfg.position_noise_clip, cfg.position_noise_clip)
        return noise

    # ── Scene loading ─────────────────────────────────────────────────────────

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg: LiftMultiShapeRandomizationConfig = self.domain_randomization_config

        def mid(lo, hi):
            return (lo + hi) / 2.0

        # ── Shape assignment ──────────────────────────────────────────────────
        # Each env gets one fixed shape for the entire training run.
        # Cycling ensures balanced coverage when num_envs is not divisible by 4.
        if self.domain_randomization:
            raw = self._batched_episode_rng.uniform(low=0.0, high=1.0)
            shape_indices = np.floor(raw * NUM_SHAPES).astype(int).clip(0, NUM_SHAPES - 1)
        else:
            shape_indices = np.arange(self.num_envs) % NUM_SHAPES

        self.item_shape_ids = common.to_tensor(shape_indices, device=self.device).long()

        # ── Per-env random dimensions (generated for all envs, masked per shape) ──
        if self.domain_randomization:
            cube_sizes  = self._batched_episode_rng.uniform(low=cfg.cube_half_size_range[0],      high=cfg.cube_half_size_range[1])
            sph_radii   = self._batched_episode_rng.uniform(low=cfg.sphere_radius_range[0],        high=cfg.sphere_radius_range[1])
            cap_radii   = self._batched_episode_rng.uniform(low=cfg.capsule_radius_range[0],       high=cfg.capsule_radius_range[1])
            cap_hls     = self._batched_episode_rng.uniform(low=cfg.capsule_half_length_range[0],  high=cfg.capsule_half_length_range[1])
            cyl_radii   = self._batched_episode_rng.uniform(low=cfg.can_radius_range[0],           high=cfg.can_radius_range[1])
            cyl_hls     = self._batched_episode_rng.uniform(low=cfg.can_half_height_range[0],      high=cfg.can_half_height_range[1])
            frictions   = self._batched_episode_rng.uniform(low=cfg.item_friction_range[0],        high=cfg.item_friction_range[1])
            densities   = self._batched_episode_rng.uniform(low=cfg.item_density_range[0],         high=cfg.item_density_range[1])
        else:
            cube_sizes  = np.full(self.num_envs, mid(*cfg.cube_half_size_range))
            sph_radii   = np.full(self.num_envs, mid(*cfg.sphere_radius_range))
            cap_radii   = np.full(self.num_envs, mid(*cfg.capsule_radius_range))
            cap_hls     = np.full(self.num_envs, mid(*cfg.capsule_half_length_range))
            cyl_radii   = np.full(self.num_envs, mid(*cfg.can_radius_range))
            cyl_hls     = np.full(self.num_envs, mid(*cfg.can_half_height_range))
            frictions   = np.full(self.num_envs, mid(*cfg.item_friction_range))
            densities   = np.full(self.num_envs, mid(*cfg.item_density_range))

        # ── Per-shape masks ───────────────────────────────────────────────────
        cube_m = shape_indices == SHAPE_CUBE
        sph_m  = shape_indices == SHAPE_SPHERE
        cap_m  = shape_indices == SHAPE_CAPSULE
        cyl_m  = shape_indices == SHAPE_CYLINDER

        # ── Effective half-height (for z-placement and lifted check) ──────────
        # item_half_sizes[i] = height of object's centre above table surface
        half_sizes = np.zeros(self.num_envs)
        half_sizes[cube_m] = cube_sizes[cube_m]
        half_sizes[sph_m]  = sph_radii[sph_m]
        half_sizes[cap_m]  = cap_radii[cap_m] + cap_hls[cap_m]   # radius + half_length
        half_sizes[cyl_m]  = cyl_hls[cyl_m]

        # ── item_dimensions: [x_extent, y_extent, z_extent] ──────────────────
        dimensions = np.zeros((self.num_envs, 3))
        dimensions[cube_m] = np.stack([cube_sizes[cube_m]] * 3, axis=-1)
        dimensions[sph_m]  = np.stack([sph_radii[sph_m]] * 3, axis=-1)
        dimensions[cap_m]  = np.stack([cap_radii[cap_m], cap_radii[cap_m], cap_radii[cap_m] + cap_hls[cap_m]], axis=-1)
        dimensions[cyl_m]  = np.stack([cyl_radii[cyl_m], cyl_radii[cyl_m], cyl_hls[cyl_m]], axis=-1)

        # ── Default colours per shape (overridden if randomize_item_color) ───
        colors = np.zeros((self.num_envs, 4))
        colors[:, 3] = 1.0
        colors[cube_m] = [0.95, 0.25, 0.25, 1.0]   # red
        colors[sph_m]  = [0.25, 0.85, 0.25, 1.0]   # green
        colors[cap_m]  = [0.95, 0.85, 0.10, 1.0]   # yellow
        colors[cyl_m]  = [0.25, 0.45, 0.95, 1.0]   # blue

        if self.domain_randomization and cfg.randomize_item_color:
            random_rgb = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
            colors = np.concatenate([random_rgb, np.ones((self.num_envs, 1))], axis=-1)

        # ── Store tensors ─────────────────────────────────────────────────────
        self.item_half_sizes  = common.to_tensor(half_sizes,  device=self.device)
        self.item_dimensions  = common.to_tensor(dimensions,  device=self.device)
        self.item_frictions   = common.to_tensor(frictions,   device=self.device)
        self.item_densities   = common.to_tensor(densities,   device=self.device)

        # ── Build actors ──────────────────────────────────────────────────────
        upright = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))   # rotates X-axis primitive to stand along Z

        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            mat = sapien.pysapien.physx.PhysxMaterial(
                static_friction=frictions[i],
                dynamic_friction=frictions[i],
                restitution=0,
            )
            render_mat = sapien.render.RenderMaterial(base_color=colors[i])
            shape = shape_indices[i]

            if shape == SHAPE_CUBE:
                hs = cube_sizes[i]
                builder.add_box_collision(half_size=[hs] * 3, material=mat, density=densities[i])
                builder.add_box_visual(half_size=[hs] * 3, material=render_mat)
                builder.initial_pose = sapien.Pose(p=[0, 0, hs])

            elif shape == SHAPE_SPHERE:
                r = sph_radii[i]
                builder.add_sphere_collision(radius=r, material=mat, density=densities[i])
                builder.add_sphere_visual(radius=r, material=render_mat)
                builder.initial_pose = sapien.Pose(p=[0, 0, r])

            elif shape == SHAPE_CAPSULE:
                r, hl = cap_radii[i], cap_hls[i]
                builder.add_capsule_collision(radius=r, half_length=hl, material=mat, density=densities[i], pose=upright)
                builder.add_capsule_visual(radius=r, half_length=hl, material=render_mat, pose=upright)
                builder.initial_pose = sapien.Pose(p=[0, 0, r + hl])

            elif shape == SHAPE_CYLINDER:
                r, hl = cyl_radii[i], cyl_hls[i]
                builder.add_cylinder_collision(radius=r, half_length=hl, material=mat, density=densities[i], pose=upright)
                builder.add_cylinder_visual(radius=r, half_length=hl, material=render_mat, pose=upright)
                builder.initial_pose = sapien.Pose(p=[0, 0, hl])

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        self._load_camera_mount()
        self._randomize_robot_color()

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict):
        obs = {}
        if self.obs_mode_struct.state:
            b = self.item.pose.p.shape[0]
            object_pos_noisy = self.item.pose.p.clone() + self._position_noise_tensor(b)

            cfg: LiftMultiShapeRandomizationConfig = self.domain_randomization_config
            if cfg.position_dropout_prob > 0:
                keep = (torch.rand((b, 1), device=self.device) >= cfg.position_dropout_prob).float()
            else:
                keep = torch.ones((b, 1), device=self.device)

            object_pos_est    = object_pos_noisy * keep
            tcp_to_object_pos = (object_pos_noisy - self.agent.tcp_pos) * keep

            shape_one_hot = torch.zeros((b, NUM_SHAPES), device=self.device)
            shape_one_hot.scatter_(1, self.item_shape_ids.unsqueeze(1), 1.0)

            # DR fields first (same convention as LiftPosition)
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

            # position features are always last (deploy injection convention)
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pos=self.agent.tcp_pose.raw_pose,
                dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1] - self.rest_qpos[:-1],
                item_shape_id=shape_one_hot,
                object_pos_est=object_pos_est,
                tcp_to_object_pos=tcp_to_object_pos,
                position_valid=keep,
            )
        return obs


@register_env("SO101LiftMultiShape-v1", max_episode_steps=100)
class LiftMultiShapeEnv(LiftMultiShape):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)