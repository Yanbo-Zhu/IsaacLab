# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv


def _validate_z_axis_align_config(name: str, scale: float, alpha_range: list[float], kappa: float) -> None:
    """Validate the PegInsert-only penalty and tolerance range [deg]."""
    if not math.isfinite(scale) or scale > 0:
        raise ValueError("z_axis_align_scale must be finite and nonpositive.")
    if name != "peg_insert" and scale != 0:
        raise ValueError("Nonzero z_axis_align_scale is only supported for PegInsert.")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("z_axis_align_kappa must be finite and positive.")
    if len(alpha_range) != 2 or not all(math.isfinite(value) for value in alpha_range):
        raise ValueError("z_axis_align_alpha_range must contain two finite angles [deg].")
    lower, upper = alpha_range
    if not 0 <= lower <= upper < 180:
        raise ValueError("z_axis_align_alpha_range must satisfy 0 <= lower <= upper < 180 [deg].")
    if not torch.tensor(math.pi, dtype=torch.float32) > torch.tensor(math.radians(upper), dtype=torch.float32):
        raise ValueError("z_axis_align_alpha_range rounds to pi in the calculation dtype.")


def _z_axis_angle(held_quat: torch.Tensor, fixed_quat: torch.Tensor) -> torch.Tensor:
    """Return the angle [rad] between asset local +Z axes using XYZW quaternions."""
    dtype = torch.float64 if torch.float64 in (held_quat.dtype, fixed_quat.dtype) else torch.float32
    axes = []
    for quat in (held_quat, fixed_quat):
        quat = quat.to(dtype=dtype)
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if not torch.all(torch.isfinite(quat)) or not torch.all(torch.isfinite(norm) & (norm > 0)):
            raise ValueError("Z-axis alignment requires finite, nonzero asset quaternions.")
        local_z = torch.zeros_like(quat[..., :3])
        local_z[..., 2] = 1.0
        axes.append(quat_apply(quat / norm, local_z))
    cosine = (axes[0] * axes[1]).sum(dim=-1).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(torch.linalg.cross(axes[0], axes[1], dim=-1), dim=-1)
    return torch.atan2(sine, cosine)


def _z_axis_penalty(theta: torch.Tensor, alpha: torch.Tensor, kappa: float) -> torch.Tensor:
    """Return the nonnegative tolerance penalty for angles and tolerances [rad]."""
    theta, alpha = torch.broadcast_tensors(theta, alpha)
    penalty = torch.zeros_like(theta)
    exceeded = theta > alpha
    # Evaluate only the valid branch. Float64 also keeps very large finite kappa
    # from becoming inf before multiplication by the zero exponent at theta=pi.
    x = ((theta[exceeded] - alpha[exceeded]) / (math.pi - alpha[exceeded])).double()
    penalty[exceeded] = torch.exp(kappa * (1.0 - x.pow(-2.0 / 3.0))).to(theta.dtype)
    return penalty


class _ZAxisAlignRandomization:
    """Sample episode tolerances [rad] without advancing the global Torch RNG."""

    def __init__(
        self, num_envs: int, device: torch.device | str, alpha_range: list[float], seed: int, rank: int
    ) -> None:
        self.alpha = torch.zeros(num_envs, device=device)
        self._bounds = tuple(math.radians(value) for value in alpha_range)
        self._generator = torch.Generator(device=device)
        self._rank = rank
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Restart the independent stream, recording its actual rank-specific seed."""
        self.seed = (seed + self._rank) % (2**64)
        self._generator.manual_seed(self.seed)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample tolerances [rad] only for the specified new episodes."""
        lower, upper = self._bounds
        if lower == upper:
            self.alpha[env_ids] = lower
        else:
            samples = torch.rand(len(env_ids), device=self.alpha.device, generator=self._generator)
            self.alpha[env_ids] = lower + (upper - lower) * samples


class _EpisodeZAxisAlignStatistics:
    """Track one angle sample per normal control step, including the terminal step."""

    def __init__(self, num_envs: int, max_steps: int, device: torch.device | str, scale: float) -> None:
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._angles = torch.zeros((num_envs, max_steps), device=device)
        self._sums = torch.zeros((num_envs, 7), device=device)
        self._scale = scale

    def update(self, theta: torch.Tensor, penalty: torch.Tensor, reference: torch.Tensor, alpha: torch.Tensor) -> None:
        """Include angles/tolerances [rad] and their penalties from this control step."""
        if torch.any(self.count >= self._angles.shape[1]):
            raise RuntimeError("Z-axis episode statistics exceeded the configured episode capacity.")
        self._angles[torch.arange(len(self.count), device=self.count.device), self.count] = theta
        self._sums += torch.stack(
            (
                theta,
                penalty,
                reference,
                (theta <= math.radians(5.0)).float(),
                (theta <= math.radians(15.0)).float(),
                (theta > alpha).float(),
                (theta - alpha).clamp_min(0.0),
            ),
            dim=-1,
        )
        self.count += 1

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear only episodes being reset, without changing an existing report."""
        self.count[env_ids] = 0
        self._angles[env_ids] = 0.0
        self._sums[env_ids] = 0.0

    def report(self, env_ids: torch.Tensor, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return independent per-episode values; reported physical angles use [deg]."""
        counts = self.count[env_ids]
        if torch.any(counts == 0):
            raise ValueError("Cannot report Z-axis statistics for an episode without samples.")
        means = self._sums[env_ids] / counts[:, None]
        angles = self._angles[env_ids]
        valid = torch.arange(angles.shape[1], device=angles.device)[None, :] < counts[:, None]
        p95 = torch.nanquantile(angles.masked_fill(~valid, float("nan")), 0.95, dim=1, interpolation="linear")
        return {
            "z_axis_align_reward_mean": means[:, 1],
            "z_axis_align_weighted_mean": self._scale * means[:, 1],
            "z_axis_align_weighted_return": self._scale * self._sums[env_ids, 1],
            "z_axis_angle_mean_deg": torch.rad2deg(means[:, 0]),
            "z_axis_angle_p95_deg": torch.rad2deg(p95),
            "z_axis_angle_final_deg": torch.rad2deg(
                angles[torch.arange(len(env_ids), device=angles.device), counts - 1]
            ),
            "z_axis_aligned_fraction_5deg": means[:, 3],
            "z_axis_align_reference_15deg_reward_mean": means[:, 2],
            "z_axis_align_alpha_deg": torch.rad2deg(alpha[env_ids]).clone(),
            "z_axis_violation_fraction": means[:, 5],
            "z_axis_excess_mean_deg": torch.rad2deg(means[:, 6]),
            "z_axis_angle_max_deg": torch.rad2deg(angles.max(dim=1).values),
            "z_axis_aligned_fraction_15deg": means[:, 4],
        }


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
