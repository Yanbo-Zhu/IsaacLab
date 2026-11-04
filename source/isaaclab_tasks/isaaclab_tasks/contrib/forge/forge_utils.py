# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv


def _validate_z_axis_alignment_config(task_name: str, alpha_deg: float, scale: float) -> None:
    """Validate the fixed tolerance [deg] and signed penalty weight."""
    if not math.isfinite(alpha_deg) or not 0.0 <= alpha_deg < 180.0:
        raise ValueError("z_axis_align_alpha_deg must be finite and in [0, 180).")
    if not math.isfinite(scale) or not -1.0 <= scale <= 0.0:
        raise ValueError("z_axis_align_scale must be finite and in [-1, 0].")
    if task_name != "peg_insert" and scale != 0.0:
        raise ValueError("The Z-axis alignment penalty is supported only for FORGE PegInsert.")


def _z_axis_alignment_penalty(theta: torch.Tensor, alpha_rad: float) -> torch.Tensor:
    """Return the cube-root penalty for axis angles [rad], shape [N]."""
    return ((theta - alpha_rad) / (math.pi - alpha_rad)).clamp(0.0, 1.0).pow(1.0 / 3.0)


def _compute_z_axis_alignment(
    held_quat: torch.Tensor, fixed_quat: torch.Tensor, alpha_rad: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local +Z angles [rad] and penalties from world XYZW quaternions."""
    local_z = torch.zeros_like(held_quat[..., :3])
    local_z[..., 2] = 1.0
    held_z = torch.nn.functional.normalize(quat_apply(held_quat, local_z), dim=-1)
    fixed_z = torch.nn.functional.normalize(quat_apply(fixed_quat, local_z), dim=-1)
    cosine = (held_z * fixed_z).sum(dim=-1).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(torch.cross(held_z, fixed_z, dim=-1), dim=-1)
    theta = torch.atan2(sine, cosine)
    return theta, _z_axis_alignment_penalty(theta, alpha_rad)


class _EpisodeZAxisAlignmentStatistics:
    """Track per-environment penalties and angle histories without reset simulation samples."""

    def __init__(
        self, num_envs: int, max_steps: int, alpha_deg: float, scale: float, device: torch.device | str
    ) -> None:
        self._alpha_deg = alpha_deg
        self._alpha_rad = math.radians(alpha_deg)
        self._scale = scale
        self._angles = torch.full((num_envs, max(1, max_steps)), torch.nan, device=device)
        self._counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._penalty_sum = torch.zeros(num_envs, device=device)
        self._env_ids = torch.arange(num_envs, device=device)
        self._step_upper_bound = 0

    def update(self, theta: torch.Tensor, penalty: torch.Tensor) -> None:
        """Include one normal step's angles [rad] and positive penalties."""
        if self._step_upper_bound >= self._angles.shape[1]:
            self._step_upper_bound = int(self._counts.max().item())
            if self._step_upper_bound >= self._angles.shape[1]:
                self._angles = torch.cat((self._angles, torch.full_like(self._angles, torch.nan)), dim=1)
        self._angles[self._env_ids, self._counts] = theta.detach()
        self._counts += 1
        self._penalty_sum += penalty.detach()
        self._step_upper_bound += 1

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear only the environments starting a new episode."""
        self._angles[env_ids] = torch.nan
        self._counts[env_ids] = 0
        self._penalty_sum[env_ids] = 0.0
        if env_ids.numel() == self._counts.numel():
            self._step_upper_bound = 0

    def report(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return independent completed-episode metrics, with angles in [deg]."""
        counts = self._counts[env_ids]
        angles = self._angles[env_ids]
        degrees = torch.rad2deg(angles)
        penalty_mean = self._penalty_sum[env_ids] / counts
        metrics = {
            "z_axis_align_penalty_mean": penalty_mean,
            "z_axis_align_weighted_mean": self._scale * penalty_mean,
            "z_axis_align_weighted_return": self._scale * self._penalty_sum[env_ids],
            "z_axis_angle_mean_deg": degrees.nanmean(dim=1),
            "z_axis_angle_p95_deg": torch.nanquantile(degrees, 0.95, dim=1),
            "z_axis_angle_final_deg": degrees.gather(1, (counts - 1).unsqueeze(1)).squeeze(1),
            "z_axis_angle_max_deg": torch.where(degrees.isnan(), -torch.inf, degrees).amax(dim=1),
            "z_axis_aligned_fraction_5deg": (angles <= math.radians(5.0)).sum(dim=1) / counts,
            "z_axis_aligned_fraction_15deg": (angles <= math.radians(15.0)).sum(dim=1) / counts,
            "z_axis_violation_fraction": (angles > self._alpha_rad).sum(dim=1) / counts,
            "z_axis_excess_mean_deg": (degrees - self._alpha_deg).clamp_min(0.0).nansum(dim=1) / counts,
            "z_axis_reverse_fraction": (angles > math.pi / 2).sum(dim=1) / counts,
            "z_axis_align_alpha_deg": torch.full_like(penalty_mean, self._alpha_deg),
        }
        return {f"ZAxisAlign/{name}": value.detach().clone() for name, value in metrics.items()}


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
