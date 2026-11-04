# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FORGE local-Z tolerance penalty and episode statistics."""

import math

import pytest
import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from isaaclab_tasks.contrib.forge.forge_utils import (
    _compute_z_axis_alignment,
    _EpisodeZAxisAlignmentStatistics,
    _validate_z_axis_alignment_config,
    _z_axis_alignment_penalty,
)


def test_penalty_has_a_dead_zone_and_monotone_cube_root_shape() -> None:
    """The tolerance is unpenalized and every larger angle increases the penalty."""
    degrees = torch.tensor([0.0, 5.0, 5.001, 15.0, 30.0, 45.0, 90.0, 180.0], dtype=torch.float64)
    penalty = _z_axis_alignment_penalty(torch.deg2rad(degrees), math.radians(5.0))
    reference = torch.tensor(
        [math.cbrt(max(float(angle) - 5.0, 0.0) / 175.0) for angle in degrees], dtype=torch.float64
    )
    torch.testing.assert_close(penalty, reference)
    assert torch.equal(penalty[:2], torch.zeros(2, dtype=torch.float64))
    assert torch.all(penalty[1:] >= penalty[:-1])
    assert penalty[-1] == 1.0
    assert torch.all((-penalty >= -1.0) & (-penalty <= 0.0))


def test_alignment_uses_world_local_z_directions() -> None:
    """Common rotation, local-Z spin and quaternion signs preserve the angle."""
    zeros = torch.zeros(4, dtype=torch.float64)
    angles = torch.deg2rad(torch.tensor([0.0, 30.0, 90.0, 180.0], dtype=torch.float64))
    peg = quat_from_euler_xyz(angles, zeros, zeros)
    socket = quat_from_euler_xyz(zeros, zeros, zeros)
    theta, penalty = _compute_z_axis_alignment(peg, socket, math.radians(5.0))
    torch.testing.assert_close(theta, angles)
    common = quat_from_euler_xyz(zeros + 0.3, zeros - 0.7, zeros + 0.2)
    spin = quat_from_euler_xyz(zeros, zeros, zeros + 1.2)
    rotated_theta, rotated_penalty = _compute_z_axis_alignment(
        -quat_mul(common, quat_mul(peg, spin)), quat_mul(common, socket), math.radians(5.0)
    )
    torch.testing.assert_close(rotated_theta, theta)
    torch.testing.assert_close(rotated_penalty, penalty)


@pytest.mark.parametrize(
    "task,alpha,scale",
    [
        ("peg_insert", -1.0, -1.0),
        ("peg_insert", 180.0, -1.0),
        ("peg_insert", float("nan"), 0.0),
        ("peg_insert", 5.0, 0.1),
        ("peg_insert", 5.0, -1.1),
        ("peg_insert", 5.0, float("inf")),
        ("gear_mesh", 5.0, -1.0),
    ],
)
def test_invalid_alignment_configuration_is_rejected(task: str, alpha: float, scale: float) -> None:
    with pytest.raises(ValueError):
        _validate_z_axis_alignment_config(task, alpha, scale)


def test_episode_statistics_include_all_steps_and_reset_independently() -> None:
    """Reports preserve per-episode values, including the last sample and its P95."""
    statistics = _EpisodeZAxisAlignmentStatistics(2, 6, 5.0, -1.0, "cpu")
    sequence = [0.0, 5.0, 15.0, 30.0, 45.0, 180.0]
    for angle in sequence:
        theta = torch.deg2rad(torch.tensor([angle, 0.0]))
        statistics.update(theta, _z_axis_alignment_penalty(theta, math.radians(5.0)))
    report = statistics.report(torch.tensor([0, 1]))
    assert len(report) == 13
    assert all(name.startswith("ZAxisAlign/") for name in report)
    expected = sum(math.cbrt(max(angle - 5.0, 0.0) / 175.0) for angle in sequence)
    torch.testing.assert_close(report["ZAxisAlign/z_axis_align_penalty_mean"], torch.tensor([expected / 6, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_align_weighted_return"], torch.tensor([-expected, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_angle_p95_deg"], torch.tensor([146.25, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_angle_final_deg"], torch.tensor([180.0, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_aligned_fraction_5deg"], torch.tensor([2 / 6, 1.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_aligned_fraction_15deg"], torch.tensor([3 / 6, 1.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_violation_fraction"], torch.tensor([4 / 6, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_reverse_fraction"], torch.tensor([1 / 6, 0.0]))
    torch.testing.assert_close(report["ZAxisAlign/z_axis_excess_mean_deg"], torch.tensor([250 / 6, 0.0]))
    statistics.reset(torch.tensor([0]))
    theta = torch.tensor([0.0, math.pi])
    statistics.update(theta, _z_axis_alignment_penalty(theta, math.radians(5.0)))
    new_report = statistics.report(torch.tensor([0, 1]))
    assert new_report["ZAxisAlign/z_axis_angle_max_deg"][0] == 0.0
    assert report["ZAxisAlign/z_axis_angle_final_deg"][0] == 180.0
    torch.testing.assert_close(new_report["ZAxisAlign/z_axis_align_penalty_mean"][1], torch.tensor(1 / 7))


def test_zero_scale_keeps_unweighted_diagnostics() -> None:
    statistics = _EpisodeZAxisAlignmentStatistics(1, 1, 5.0, 0.0, "cpu")
    statistics.update(torch.tensor([math.pi]), torch.ones(1))
    report = statistics.report(torch.tensor([0]))
    assert report["ZAxisAlign/z_axis_align_penalty_mean"].item() == 1.0
    assert report["ZAxisAlign/z_axis_align_weighted_mean"].item() == 0.0
    assert report["ZAxisAlign/z_axis_align_weighted_return"].item() == 0.0


def test_hydra_overrides_preserve_policy_contract_and_other_task_defaults() -> None:
    """The new fields compose through the existing CLI without extra policy inputs."""
    from isaaclab_tasks.contrib.forge.forge_env_cfg import (
        OBS_DIM_CFG,
        STATE_DIM_CFG,
        ForgeTaskGearMeshCfg,
        ForgeTaskNutThreadCfg,
    )
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config(
        "IsaacContrib-Forge-PegInsert-Direct",
        "rl_games_cfg_entry_point",
        overrides=["env.task.z_axis_align_alpha_deg=10.0", "env.task.z_axis_align_scale=-0.25"],
    )
    assert cfg.task.z_axis_align_alpha_deg == 10.0
    assert cfg.task.z_axis_align_scale == -0.25
    assert sum(OBS_DIM_CFG[name] for name in cfg.obs_order) + cfg.action_space == 24
    assert sum(STATE_DIM_CFG[name] for name in cfg.state_order) + cfg.action_space == 61
    assert cfg.action_space == 7
    for other in (ForgeTaskGearMeshCfg(), ForgeTaskNutThreadCfg()):
        assert other.task.z_axis_align_scale == 0.0
        _validate_z_axis_alignment_config(other.task.name, other.task.z_axis_align_alpha_deg, 0.0)
