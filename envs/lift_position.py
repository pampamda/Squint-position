from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from mani_skill.utils import common
from mani_skill.utils.registration import register_env

from .lift import Lift, LiftRandomizationConfig


@dataclass
class LiftPositionRandomizationConfig(LiftRandomizationConfig):
    """Lift position-estimation settings for sim-GT-with-noise MVP."""

    position_noise_std: Sequence[float] = (0.005, 0.005, 0.003)
    position_noise_clip: float = 0.02
    position_dropout_prob: float = 0.0


class LiftPosition(Lift):
    """Lift task variant exposing explicit object position observations."""

    def __init__(
        self,
        *args,
        item_type: str = "cube",
        domain_randomization_config: Union[
            LiftPositionRandomizationConfig, dict
        ] = LiftPositionRandomizationConfig(),
        **kwargs,
    ):
        if item_type != "cube":
            raise NotImplementedError("LiftPosition MVP currently supports cube only.")
        super().__init__(
            *args,
            item_type=item_type,
            domain_randomization_config=domain_randomization_config,
            **kwargs,
        )

    def _position_noise_tensor(self, b: int) -> torch.Tensor:
        cfg: LiftPositionRandomizationConfig = self.domain_randomization_config
        std = torch.tensor(cfg.position_noise_std, device=self.device, dtype=torch.float32)
        noise = torch.randn((b, 3), device=self.device) * std
        if cfg.position_noise_clip > 0:
            noise = torch.clamp(noise, -cfg.position_noise_clip, cfg.position_noise_clip)
        return noise

    def _get_obs_extra(self, info: dict):
        obs = {}
        if self.obs_mode_struct.state:
            b = self.item.pose.p.shape[0]
            object_pos_base = self.item.pose.p.clone()
            object_pos_noisy = object_pos_base + self._position_noise_tensor(b)

            cfg: Optional[LiftPositionRandomizationConfig] = self.domain_randomization_config
            if cfg.position_dropout_prob > 0:
                keep = (
                    torch.rand((b, 1), device=self.device) >= cfg.position_dropout_prob
                ).float()
            else:
                keep = torch.ones((b, 1), device=self.device)

            # valid=0 时两个位置字段都归零，保持语义一致
            # tcp_to_object_pos 必须在 keep 应用前计算，否则 valid=0 时得到 -tcp_pos
            object_pos_est = object_pos_noisy * keep
            tcp_to_object_pos = (object_pos_noisy - self.agent.tcp_pos) * keep

            # DR 字段先插入，确保位置特征始终位于 state 末尾（deploy 注入依赖此约定）
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

            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                tcp_pos=self.agent.tcp_pose.raw_pose,
                dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1]
                - self.rest_qpos[:-1],
                object_pos_est=object_pos_est,
                tcp_to_object_pos=tcp_to_object_pos,
                position_valid=keep,
            )
        return obs


@register_env("SO101LiftCubePos-v1", max_episode_steps=50)
class LiftCubePosition(LiftPosition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)
