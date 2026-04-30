from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PositionProvider(Protocol):
    def get_position(self, sim_env, real_obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (object_pos_est, tcp_to_object_pos, position_valid) for each batch.

        object_pos_est: (B, 3) 位置估计，valid=0 时为零向量
        tcp_to_object_pos: (B, 3) tcp 到物体的向量
        position_valid: (B, 1) float，1=有效，0=检测失败
        """


@dataclass
class SimGTPositionProvider:
    """MVP provider: 仿真真值位置 + 可配置噪声/dropout。

    注意：仅用于纯仿真测试。真实部署时仿真物体不跟随真实物体移动，
    返回的位置是错误的。对接真实感知算法请使用 RealDetectorPositionProvider。
    """

    noise_std_xyz: tuple[float, float, float] = (0.005, 0.005, 0.003)
    noise_clip: float = 0.02
    dropout_prob: float = 0.0

    def get_position(self, sim_env, real_obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = real_obs["state"].device
        obj = sim_env.unwrapped.item.pose.p.to(device).clone()
        tcp = sim_env.unwrapped.agent.tcp_pos.to(device).clone()

        std = torch.tensor(self.noise_std_xyz, device=device, dtype=torch.float32)
        noise = torch.randn_like(obj) * std
        if self.noise_clip > 0:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        obj_est = obj + noise

        if self.dropout_prob > 0:
            valid = (torch.rand((obj.shape[0], 1), device=device) >= self.dropout_prob).float()
        else:
            valid = torch.ones((obj.shape[0], 1), device=device)
        # valid=0 时两个字段都归零，与训练环境保持一致
        # tcp_to_obj 必须在 valid 应用前计算，否则 valid=0 时得到 -tcp
        tcp_to_obj = (obj_est - tcp) * valid
        obj_est = obj_est * valid
        return obj_est, tcp_to_obj, valid


class RealDetectorPositionProvider:
    """Placeholder for second phase: RGB-D detector based provider."""

    def get_position(self, sim_env, real_obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "RealDetectorPositionProvider is a placeholder. "
            "Implement RGB-D detection and base-frame projection in phase 2."
        )

