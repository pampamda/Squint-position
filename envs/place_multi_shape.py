from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

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

from .place import Place, PlaceRandomizationConfig

# Shape constants (same as lift_multi_shape for consistency)
SHAPE_CUBE = 0
SHAPE_SPHERE = 1
SHAPE_CAPSULE = 2
SHAPE_CYLINDER = 3
NUM_SHAPES = 4
SHAPE_NAMES = ["cube", "sphere", "capsule", "cylinder"]


@dataclass
class PlaceMultiShapeRandomizationConfig(PlaceRandomizationConfig):
    """Extends PlaceRandomizationConfig with sphere and capsule geometry ranges."""
    sphere_radius_range: Sequence[float] = (0.018, 0.030)
    capsule_radius_range: Sequence[float] = (0.012, 0.020)
    capsule_half_length_range: Sequence[float] = (0.020, 0.040)


class PlaceMultiShape(Place):
    """
    Place variant that trains on cube, sphere, capsule, and cylinder simultaneously.

    Each parallel env is assigned one fixed shape for the entire training run.
    item_shape_id (4-dim one-hot) is added to observations so the policy can
    condition its grasp and place strategy on the object type.

    evaluate() and compute_dense_reward() are inherited unchanged — they both
    use item_half_sizes (effective half-height) which is set correctly per shape.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[
            PlaceMultiShapeRandomizationConfig, dict
        ] = PlaceMultiShapeRandomizationConfig(),
        **kwargs,
    ):
        if isinstance(domain_randomization_config, dict):
            merged = asdict(PlaceMultiShapeRandomizationConfig())
            common.dict_merge(merged, domain_randomization_config)
            domain_randomization_config = dacite.from_dict(
                data_class=PlaceMultiShapeRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        # item_type="mixed" bypasses Place._load_scene()'s cube/can guard
        super().__init__(
            *args,
            item_type="mixed",
            domain_randomization_config=domain_randomization_config,
            **kwargs,
        )

    # ── Scene loading ─────────────────────────────────────────────────────────

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg: PlaceMultiShapeRandomizationConfig = self.domain_randomization_config

        def mid(lo, hi):
            return (lo + hi) / 2.0

        # ── Shape assignment ──────────────────────────────────────────────────
        if self.domain_randomization:
            raw = self._batched_episode_rng.uniform(low=0.0, high=1.0)
            shape_indices = np.floor(raw * NUM_SHAPES).astype(int).clip(0, NUM_SHAPES - 1)
        else:
            shape_indices = np.arange(self.num_envs) % NUM_SHAPES

        self.item_shape_ids = common.to_tensor(shape_indices, device=self.device).long()

        # ── Per-env random dimensions ─────────────────────────────────────────
        if self.domain_randomization:
            cube_sizes  = self._batched_episode_rng.uniform(low=cfg.cube_half_size_range[0],     high=cfg.cube_half_size_range[1])
            sph_radii   = self._batched_episode_rng.uniform(low=cfg.sphere_radius_range[0],       high=cfg.sphere_radius_range[1])
            cap_radii   = self._batched_episode_rng.uniform(low=cfg.capsule_radius_range[0],      high=cfg.capsule_radius_range[1])
            cap_hls     = self._batched_episode_rng.uniform(low=cfg.capsule_half_length_range[0], high=cfg.capsule_half_length_range[1])
            cyl_radii   = self._batched_episode_rng.uniform(low=cfg.can_radius_range[0],          high=cfg.can_radius_range[1])
            cyl_hls     = self._batched_episode_rng.uniform(low=cfg.can_half_height_range[0],     high=cfg.can_half_height_range[1])
            frictions   = self._batched_episode_rng.uniform(low=cfg.item_friction_range[0],       high=cfg.item_friction_range[1])
            densities   = self._batched_episode_rng.uniform(low=cfg.item_density_range[0],        high=cfg.item_density_range[1])
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

        # ── Effective half-height (z placement, lifted check, goal height) ────
        half_sizes = np.zeros(self.num_envs)
        half_sizes[cube_m] = cube_sizes[cube_m]
        half_sizes[sph_m]  = sph_radii[sph_m]
        half_sizes[cap_m]  = cap_radii[cap_m] + cap_hls[cap_m]
        half_sizes[cyl_m]  = cyl_hls[cyl_m]

        # ── item_dimensions [x, y, z] ─────────────────────────────────────────
        dimensions = np.zeros((self.num_envs, 3))
        dimensions[cube_m] = np.stack([cube_sizes[cube_m]] * 3, axis=-1)
        dimensions[sph_m]  = np.stack([sph_radii[sph_m]] * 3, axis=-1)
        dimensions[cap_m]  = np.stack([cap_radii[cap_m], cap_radii[cap_m], cap_radii[cap_m] + cap_hls[cap_m]], axis=-1)
        dimensions[cyl_m]  = np.stack([cyl_radii[cyl_m], cyl_radii[cyl_m], cyl_hls[cyl_m]], axis=-1)

        # ── Horizontal placement radius (for UniformPlacementSampler) ─────────
        # Cube: circumscribed circle; sphere/capsule/cylinder: their horizontal radius
        placement_radii = np.zeros(self.num_envs)
        placement_radii[cube_m] = cube_sizes[cube_m] * np.sqrt(2)
        placement_radii[sph_m]  = sph_radii[sph_m]
        placement_radii[cap_m]  = cap_radii[cap_m]
        placement_radii[cyl_m]  = cyl_radii[cyl_m]
        self.item_placement_radii = common.to_tensor(placement_radii, device=self.device)

        # ── Default colours per shape ─────────────────────────────────────────
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

        # ── Build item actors ─────────────────────────────────────────────────
        upright = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))

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
                builder.initial_pose = sapien.Pose(p=[0.2, 0, hs])

            elif shape == SHAPE_SPHERE:
                r = sph_radii[i]
                builder.add_sphere_collision(radius=r, material=mat, density=densities[i])
                builder.add_sphere_visual(radius=r, material=render_mat)
                builder.initial_pose = sapien.Pose(p=[0.2, 0, r])

            elif shape == SHAPE_CAPSULE:
                r, hl = cap_radii[i], cap_hls[i]
                builder.add_capsule_collision(radius=r, half_length=hl, material=mat, density=densities[i], pose=upright)
                builder.add_capsule_visual(radius=r, half_length=hl, material=render_mat, pose=upright)
                builder.initial_pose = sapien.Pose(p=[0.2, 0, r + hl])

            elif shape == SHAPE_CYLINDER:
                r, hl = cyl_radii[i], cyl_hls[i]
                builder.add_cylinder_collision(radius=r, half_length=hl, material=mat, density=densities[i], pose=upright)
                builder.add_cylinder_visual(radius=r, half_length=hl, material=render_mat, pose=upright)
                builder.initial_pose = sapien.Pose(p=[0.2, 0, hl])

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # ── Build bins (identical to Place._load_scene) ───────────────────────
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness

        bin_half_sizes_x = np.ones(self.num_envs) * mid(*cfg.bin_half_size_x_range)
        bin_half_sizes_y = np.ones(self.num_envs) * mid(*cfg.bin_half_size_y_range)
        bin_half_sizes_z = np.ones(self.num_envs) * mid(*cfg.bin_half_size_z_range)

        if self.domain_randomization:
            bin_half_sizes_x = self._batched_episode_rng.uniform(low=cfg.bin_half_size_x_range[0], high=cfg.bin_half_size_x_range[1])
            bin_half_sizes_y = self._batched_episode_rng.uniform(low=cfg.bin_half_size_y_range[0], high=cfg.bin_half_size_y_range[1])
            bin_half_sizes_z = self._batched_episode_rng.uniform(low=cfg.bin_half_size_z_range[0], high=cfg.bin_half_size_z_range[1])

        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions   = torch.stack([self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1)

        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

            bin_center_pose      = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

            for j in [-1, 1]:
                y = j * bin_center_half_size[1]
                wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)

                x = j * bin_center_half_size[0]
                wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, bin_half_size[2]])
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)
        self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        self._load_camera_mount()
        self._randomize_robot_color()

        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01,
            # White: matches bin colour, clearly not a graspable item.
            # Avoids confusion with green sphere items.
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    # ── Episode initialisation ────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Call DefaultCameraEnv._initialize_episode (skipping Place's which has item_type check)
        super(Place, self)._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size,  self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            # Use per-shape horizontal placement radius instead of item_type check
            item_radius = self.item_placement_radii.max().item() + 0.01
            bin_radius  = self.bin_radius.max().item() + 0.01

            item_xy_offset = sampler.sample(item_radius, 100)
            bin_xy_offset  = sampler.sample(bin_radius, 100, verbose=False)

            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = spawn_center[env_idx, :2] + item_xy_offset
            item_xyz[:, 2]  = self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2]  = self.bin_thickness / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            goal_xyz = bin_xyz.clone()
            # Clamp goal height to stay inside bin walls for tall items (capsule, cylinder)
            goal_height = torch.minimum(
                self.item_half_sizes[env_idx],
                self.bin_half_sizes_z[env_idx],
            )
            goal_xyz[:, 2] = self.bin_thickness + goal_height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.state:
            b = self.item.pose.p.shape[0]
            shape_one_hot = torch.zeros((b, NUM_SHAPES), device=self.device)
            shape_one_hot.scatter_(1, self.item_shape_ids.unsqueeze(1), 1.0)
            obs["item_shape_id"] = shape_one_hot
        return obs


@register_env("SO101PlaceMultiShape-v1", max_episode_steps=100)
class PlaceMultiShapeEnv(PlaceMultiShape):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
