# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv


def _compute_z_axis_alignment(
    held_quat: torch.Tensor, fixed_quat: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return directional rewards, angles [deg], and fixed-5-degree reference rewards.

    Args:
        held_quat: Held-asset world unit quaternions in (x, y, z, w), shape [N, 4].
        fixed_quat: Fixed-asset world unit quaternions in (x, y, z, w), shape [N, 4].
        sigma: Per-environment reward decay angles [rad], shape [N].
    """
    local_z = torch.zeros_like(held_quat[:, :3])
    local_z[:, 2] = 1.0
    held_z = quat_apply(held_quat, local_z)
    fixed_z = quat_apply(fixed_quat, local_z)
    cosine = (held_z * fixed_z).sum(dim=-1).clamp(-1.0, 1.0)
    difference = 1.0 - cosine
    denominator = 2.0 * torch.sin(sigma / 2.0).square()
    reward = torch.exp(-difference / denominator)
    reference_denominator = 2.0 * math.sin(math.radians(5.0) / 2.0) ** 2
    reference_reward = torch.exp(-difference / reference_denominator)
    return reward, torch.rad2deg(torch.acos(cosine)), reference_reward


class _ZAxisAlignmentReward:
    """Validate and sample episode-constant reward scales using an isolated RNG."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        seed: int | None,
        scale: float,
        sigma_range: list[float],
    ) -> None:
        """Initialize reward settings; ``sigma_range`` bounds are in [deg]."""
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("z_axis_align_scale must be finite and non-negative.")
        if len(sigma_range) != 2 or not all(math.isfinite(value) for value in sigma_range):
            raise ValueError("z_axis_align_sigma_range must contain two finite bounds [deg].")
        lower, upper = sigma_range
        if not 0.0 < lower <= upper <= 180.0:
            raise ValueError("z_axis_align_sigma_range must satisfy 0 < lower <= upper <= 180 [deg].")
        bounds = torch.deg2rad(torch.tensor(sigma_range, dtype=torch.float32, device=device))
        denominators = 2.0 * torch.sin(bounds / 2.0).square()
        if not torch.all(torch.isfinite(denominators) & (denominators > 0.0)):
            raise ValueError("z_axis_align_sigma_range produces a non-positive or non-finite denominator.")
        self.scale = scale
        self._lower, self._upper = bounds.unbind()
        self._fixed = lower == upper
        self.sigma = self._lower.repeat(num_envs)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(torch.initial_seed() if seed is None else seed)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample decay angles [rad] for only the environments starting a new episode."""
        if self._fixed:
            self.sigma[env_ids] = self._lower
        else:
            samples = torch.rand(env_ids.numel(), device=self.sigma.device, generator=self.generator)
            self.sigma[env_ids] = self._lower + samples * (self._upper - self._lower)

    def compute(
        self, held_quat: torch.Tensor, fixed_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rewards and angles [deg] without changing episode parameters."""
        return _compute_z_axis_alignment(held_quat, fixed_quat, self.sigma)


class _EpisodeZAxisAlignmentStatistics:
    """Track per-trajectory alignment metrics using only normal environment steps."""

    def __init__(self, num_envs: int, max_episode_length: int, device: torch.device | str) -> None:
        self._env_ids = torch.arange(num_envs, device=device)
        self._counts = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._reward_sum = torch.zeros(num_envs, device=device)
        self._reference_sum = torch.zeros(num_envs, device=device)
        self._angles = torch.full((num_envs, max_episode_length), torch.nan, device=device)

    def update(self, reward: torch.Tensor, angle_deg: torch.Tensor, reference_reward: torch.Tensor) -> None:
        """Record one reward and axis-angle sample [deg] per environment."""
        self._angles[self._env_ids, self._counts] = angle_deg
        self._counts += 1
        self._reward_sum += reward
        self._reference_sum += reference_reward

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear only the trajectories beginning a new episode."""
        self._counts[env_ids] = 0
        self._reward_sum[env_ids] = 0.0
        self._reference_sum[env_ids] = 0.0
        self._angles[env_ids] = torch.nan

    def report(self, env_ids: torch.Tensor, sigma: torch.Tensor, scale: float) -> dict[str, torch.Tensor]:
        """Return completed-episode means, using decay angles ``sigma`` [rad].

        The reported P95 is the mean of individual trajectories' P95 angles [deg],
        not a quantile over pooled trajectories. Unused history entries are NaN.
        """
        counts = self._counts[env_ids]
        angles = self._angles[env_ids]
        reward_means = self._reward_sum[env_ids] / counts
        metrics = {
            "z_axis_align_reward_mean": reward_means.mean(),
            "z_axis_align_weighted_mean": (reward_means * scale).mean(),
            "z_axis_align_weighted_return": (self._reward_sum[env_ids] * scale).mean(),
            "z_axis_angle_mean_deg": torch.nanmean(angles, dim=1).mean(),
            "z_axis_angle_p95_deg": torch.nanquantile(angles, 0.95, dim=1).mean(),
            "z_axis_angle_final_deg": self._angles[env_ids, counts - 1].mean(),
            "z_axis_aligned_fraction_5deg": ((angles <= 5.0).sum(dim=1) / counts).mean(),
            "z_axis_align_reference_5deg_mean": (self._reference_sum[env_ids] / counts).mean(),
            "z_axis_align_sigma_deg": torch.rad2deg(sigma[env_ids]).mean(),
        }
        return {f"ZAxisAlign/{name}": value for name, value in metrics.items()}


class _EpisodeForceStatistics:
    """Track force-magnitude statistics for vectorized episodes."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self.sum = torch.zeros(num_envs, device=device)
        self.maximum = torch.zeros(num_envs, device=device)

    def update(self, force_magnitude: torch.Tensor) -> None:
        """Include one force-magnitude sample from every environment."""
        self.sum += force_magnitude
        torch.maximum(self.maximum, force_magnitude, out=self.maximum)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear statistics for environments beginning a new episode."""
        self.sum[env_ids] = 0.0
        self.maximum[env_ids] = 0.0

    def report(self, episode_lengths: torch.Tensor, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return batch-averaged metrics for completed episodes."""
        force_means = self.sum[env_ids] / episode_lengths[env_ids]
        return {"F_mean": force_means.mean(), "F_max": self.maximum[env_ids].mean()}


def get_random_prop_gains(default_values, noise_levels, num_envs, device):
    """Helper function to randomize controller gains."""
    c_param_noise = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device)
    c_param_noise = c_param_noise @ torch.diag(torch.tensor(noise_levels, dtype=torch.float32, device=device))
    c_param_multiplier = 1.0 + c_param_noise
    decrease_param_flag = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device) > 0.5
    c_param_multiplier = torch.where(decrease_param_flag, 1.0 / c_param_multiplier, c_param_multiplier)

    prop_gains = default_values * c_param_multiplier

    return prop_gains


def change_FT_frame(source_F, source_T, source_frame, target_frame):
    """Convert force/torque reading from source to target frame.

    Args:
        source_F: Force in source frame.
        source_T: Torque in source frame.
        source_frame: Tuple of (quat_xyzw, pos) for source frame.
        target_frame: Tuple of (quat_xyzw, pos) for target frame.

    Returns:
        Tuple of (target_F, target_T) - force and torque in target frame.
    """
    # Modern Robotics eq. 3.95
    # Compute inverse of source frame
    source_quat_inv = quat_inv(source_frame[0])
    source_pos_inv = -quat_apply(source_quat_inv, source_frame[1])

    # Combine: source_inv * target = target_T_source
    target_T_source_pos, target_T_source_quat = combine_frame_transforms(
        source_pos_inv, source_quat_inv, target_frame[1], target_frame[0]
    )

    target_F = quat_apply(target_T_source_quat, source_F)
    target_T = quat_apply(target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1)))
    return target_F, target_T
