# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FORGE force and local-Z episode metrics."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from isaaclab_tasks.contrib.forge.forge_utils import (
    _EpisodeForceStatistics,
    _EpisodeZAxisAlignStatistics,
    _validate_z_axis_align_config,
    _z_axis_angle,
    _z_axis_penalty,
    _ZAxisAlignRandomization,
)


def test_episode_force_statistics_are_isolated_per_environment() -> None:
    """Force means and maxima use each environment's episode samples only."""
    statistics = _EpisodeForceStatistics(num_envs=2, device="cpu")
    statistics.update(torch.tensor([3.0, 6.0]))
    statistics.update(torch.tensor([4.0, 2.0]))

    completed_env_ids = torch.tensor([0, 1])
    episode_lengths = torch.tensor([2, 2])
    metrics = statistics.report(episode_lengths, completed_env_ids)

    assert metrics["F_mean"] == 3.75
    assert metrics["F_max"] == 5.0

    statistics.reset(torch.tensor([0]))

    assert torch.equal(statistics.sum, torch.tensor([0.0, 8.0]))
    assert torch.equal(statistics.maximum, torch.tensor([0.0, 6.0]))


@pytest.mark.parametrize("alpha_deg", [0.0, 15.0, 179.99])
@pytest.mark.parametrize("kappa", [1.0e-12, 0.1756542881, 1.0e300])
def test_z_axis_penalty_boundaries_and_monotonicity(alpha_deg: float, kappa: float) -> None:
    """The tolerance branch is finite and the positive penalty reaches one only at pi."""
    alpha = torch.tensor(math.radians(alpha_deg), dtype=torch.float64)
    theta = torch.linspace(alpha.item(), math.pi, 101, dtype=torch.float64)
    penalty = _z_axis_penalty(theta, alpha, kappa)
    assert penalty[0] == 0
    assert penalty[-1] == 1
    assert torch.isfinite(penalty).all()
    assert torch.all((penalty >= 0) & (penalty <= 1))
    assert torch.all(penalty[1:] >= penalty[:-1])
    assert torch.all(-penalty[1:] <= -penalty[:-1])


def test_z_axis_penalty_reference_and_tolerance_neighbors() -> None:
    """Reference calibration and both sides of the exact threshold follow the specified curve."""
    alpha = torch.tensor(math.radians(15.0))
    theta = torch.tensor([0.0, alpha.item() - 1e-6, alpha.item(), alpha.item() + 1e-6, math.radians(30.0)])
    penalty = _z_axis_penalty(theta, alpha, 0.1756542881)
    torch.testing.assert_close(penalty[:3], torch.zeros(3))
    assert torch.isfinite(penalty).all()
    assert penalty[-1].item() == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize(
    "name, scale, bounds, kappa",
    [
        ("peg_insert", 1, [10, 20], 1),
        ("peg_insert", float("nan"), [10, 20], 1),
        ("gear_mesh", -1, [10, 20], 1),
        ("peg_insert", -1, [10, 20], 0),
        ("peg_insert", -1, [10, 20], float("inf")),
        ("peg_insert", -1, [10], 1),
        ("peg_insert", -1, [20, 10], 1),
        ("peg_insert", -1, [-1, 10], 1),
        ("peg_insert", -1, [10, 180], 1),
        ("peg_insert", -1, [10, float("nan")], 1),
        ("peg_insert", -1, [10, 180 - 1e-8], 1),
    ],
)
def test_z_axis_config_rejects_invalid_values(name, scale, bounds, kappa) -> None:
    """Invalid signs, shapes, ranges and float32-unrepresentable tolerances fail clearly."""
    with pytest.raises(ValueError):
        _validate_z_axis_align_config(name, scale, bounds, kappa)


def test_z_axis_geometry_invariances_and_invalid_quaternions() -> None:
    """Use directed local Z, invariant to quaternion sign, local spin and common rotation."""
    angles = torch.deg2rad(torch.tensor([0.0, 30.0, 180.0]))
    zeros = torch.zeros(3)
    peg = quat_from_euler_xyz(angles, zeros, zeros)
    socket = quat_from_euler_xyz(zeros, zeros, zeros)
    original = peg.clone()
    torch.testing.assert_close(_z_axis_angle(peg * 2, socket), angles)
    torch.testing.assert_close(_z_axis_angle(-peg, socket), angles)
    spin = quat_from_euler_xyz(zeros, zeros, torch.tensor([0.2, 0.4, -0.5]))
    torch.testing.assert_close(_z_axis_angle(quat_mul(peg, spin), socket), angles)
    world = quat_from_euler_xyz(zeros + 0.2, zeros - 0.7, zeros + 0.6)
    torch.testing.assert_close(
        _z_axis_angle(quat_mul(world, peg), quat_mul(world, socket)), angles, atol=2e-7, rtol=1e-5
    )
    torch.testing.assert_close(peg, original)
    assert _z_axis_angle(peg.half(), socket.half()).dtype == torch.float32
    for invalid in (torch.zeros_like(peg), torch.full_like(peg, float("nan"))):
        with pytest.raises(ValueError, match="quaternions"):
            _z_axis_angle(invalid, socket)


def test_z_axis_rng_is_independent_replayable_and_locally_reset() -> None:
    """Episode sampling is isolated from the global stream and respects reset IDs."""
    state = torch.random.get_rng_state().clone()
    randomization = _ZAxisAlignRandomization(4, "cpu", [10, 20], seed=123, rank=2)
    ids = torch.arange(4)
    randomization.reset(ids)
    first = randomization.alpha.clone()
    randomization.reset(torch.tensor([1]))
    torch.testing.assert_close(randomization.alpha[[0, 2, 3]], first[[0, 2, 3]])
    randomization.reseed(123)
    randomization.reset(ids)
    torch.testing.assert_close(randomization.alpha, first)
    assert randomization.seed == 125
    assert torch.equal(torch.random.get_rng_state(), state)
    fixed = _ZAxisAlignRandomization(4, "cpu", [15, 15], seed=123, rank=0)
    stream_state = fixed._generator.get_state().clone()
    fixed.reset(ids)
    assert torch.equal(fixed._generator.get_state(), stream_state)
    torch.testing.assert_close(fixed.alpha, torch.full((4,), math.radians(15)))


def test_z_axis_episode_statistics_include_final_step_and_own_report_data() -> None:
    """Independently calculable trajectories verify all 13 metrics and reset isolation."""
    statistics = _EpisodeZAxisAlignStatistics(2, 8, "cpu", scale=-1)
    alpha = torch.deg2rad(torch.tensor([15.0, 20.0]))
    for angle in (0, 10, 20, 30):
        theta = torch.deg2rad(torch.tensor([float(angle), 0.0]))
        penalty = _z_axis_penalty(theta, alpha, 0.1756542881)
        reference = _z_axis_penalty(theta, alpha[:1], 0.1756542881)
        statistics.update(theta, penalty, reference, alpha)
    metrics = statistics.report(torch.arange(2), alpha)
    assert len(metrics) == 13
    expected = {
        "z_axis_angle_mean_deg": 15,
        "z_axis_angle_p95_deg": 28.5,
        "z_axis_angle_final_deg": 30,
        "z_axis_angle_max_deg": 30,
        "z_axis_aligned_fraction_5deg": 0.25,
        "z_axis_aligned_fraction_15deg": 0.5,
        "z_axis_violation_fraction": 0.5,
        "z_axis_excess_mean_deg": 5,
        "z_axis_align_alpha_deg": 15,
    }
    for name, value in expected.items():
        assert metrics[name][0].item() == pytest.approx(value, abs=1e-5)
    # Independently evaluate the two nonzero samples with scalar standard-library math.
    reference_mean = sum(math.exp(0.1756542881 * (1 - ((angle - 15) / 165) ** (-2 / 3))) for angle in (20, 30)) / 4
    assert metrics["z_axis_align_reward_mean"][0].item() == pytest.approx(reference_mean)
    assert metrics["z_axis_align_reference_15deg_reward_mean"][0].item() == pytest.approx(reference_mean)
    assert metrics["z_axis_align_weighted_mean"][0].item() == pytest.approx(-reference_mean)
    assert metrics["z_axis_align_weighted_return"][0].item() == pytest.approx(-4 * reference_mean)
    statistics.reset(torch.tensor([0]))
    alpha[0] = 0
    assert statistics.count.tolist() == [0, 4]
    assert metrics["z_axis_align_alpha_deg"][0].item() == pytest.approx(15)
    assert metrics["z_axis_angle_final_deg"][0].item() == pytest.approx(30, abs=1e-5)
    with pytest.raises(ValueError, match="without samples"):
        statistics.report(torch.tensor([0]), alpha)
    theta = torch.deg2rad(torch.tensor([90.0, 20.0]))
    penalty = _z_axis_penalty(theta, alpha, 0.1756542881)
    statistics.update(theta, penalty, penalty, alpha)
    next_metrics = statistics.report(torch.arange(2), alpha)
    torch.testing.assert_close(next_metrics["z_axis_angle_p95_deg"], torch.tensor([90.0, 16.0]))
    assert statistics.count.tolist() == [1, 5]


def test_z_axis_reward_integration_preserves_zero_scale_and_reports_terminal_once(monkeypatch) -> None:
    """Exercise the actual reward hook at identical poses/actions with and without its scale."""
    from isaaclab_tasks.contrib.factory.factory_env import FactoryEnv
    from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv

    monkeypatch.setattr(FactoryEnv, "_get_rewards", lambda _self: torch.tensor([2.0]))
    env = ForgeEnv.__new__(ForgeEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=1)
    env.cfg = SimpleNamespace(ctrl=SimpleNamespace(pos_action_threshold=[1.0], rot_action_threshold=[1.0]))
    env.cfg_task = SimpleNamespace(
        name="peg_insert",
        success_threshold=0.01,
        delay_until_ratio=1.0,
        action_penalty_asset_scale=0.0,
        contact_penalty_scale=0.0,
        z_axis_align_scale=0.0,
        z_axis_align_kappa=0.1756542881,
    )
    env._z_axis_align_randomization = _ZAxisAlignRandomization(1, "cpu", [15, 15], 0, 0)
    env._z_axis_align_randomization.reset(torch.tensor([0]))
    env._z_axis_align_statistics = _EpisodeZAxisAlignStatistics(1, 4, "cpu", 0.0)
    env._z_axis_align_reference_alpha = torch.tensor(math.radians(15))
    env._episode_force_statistics = _EpisodeForceStatistics(1, "cpu")
    env.held_quat = quat_from_euler_xyz(torch.tensor([math.radians(30)]), torch.zeros(1), torch.zeros(1))
    env.fixed_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    env.delta_pos = torch.zeros((1, 3))
    env.delta_yaw = torch.zeros(1)
    env.force_sensor_smooth = torch.zeros((1, 6))
    env.contact_penalty_thresholds = torch.ones(1)
    env.actions = torch.zeros((1, 7))
    env.success_pred_scale = 0.0
    env.reset_buf = torch.tensor([True])
    env.extras = {}
    env._get_curr_successes = lambda **_kwargs: torch.zeros(1, dtype=torch.bool)
    env._log_forge_metrics = lambda rewards, _pred: env.extras.update(
        {f"logs_rew_{name}": value.mean() for name, value in rewards.items()}
    )
    assert env._get_rewards().item() == 2.0
    payload = env.extras["z_axis_align"]
    assert env.extras["logs_rew_z_axis_align"].item() == pytest.approx(0.5, abs=1e-6)
    assert payload["metrics"]["z_axis_align_weighted_return"].item() == 0.0
    env._z_axis_align_statistics.reset(torch.tensor([0]))
    assert payload["metrics"]["z_axis_angle_final_deg"].item() == pytest.approx(30, abs=1e-5)
    env.cfg_task.z_axis_align_scale = -1.0
    env.reset_buf[:] = False
    assert env._get_rewards().item() == pytest.approx(1.5, abs=1e-6)
    assert "z_axis_align" not in env.extras
    assert env._z_axis_align_statistics.count.item() == 1


def test_z_axis_explicit_reset_reseeds_before_parent_once(monkeypatch) -> None:
    """The instance reset supports replay without a second global seed call in its parent."""
    from isaaclab_tasks.contrib.factory.factory_env import FactoryEnv
    from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv

    env = ForgeEnv.__new__(ForgeEnv)
    env._is_closed = True
    env.extras = {"z_axis_align": {}}
    env._z_axis_align_randomization = _ZAxisAlignRandomization(2, "cpu", [10, 20], 1, 0)
    seeds = []
    env.seed = lambda value: seeds.append(value) or value

    def parent_reset(_self, seed=None, options=None):
        assert seed is None
        assert "z_axis_align" not in env.extras
        env._z_axis_align_randomization.reset(torch.arange(2))
        return env._z_axis_align_randomization.alpha.clone(), {}

    monkeypatch.setattr(FactoryEnv, "reset", parent_reset)
    first, _ = env.reset(seed=42)
    env.reset()
    replay, _ = env.reset(seed=42)
    torch.testing.assert_close(first, replay)
    assert seeds == [42, 42]


def test_z_axis_defaults_preserve_ppo_interfaces() -> None:
    """The new reward keeps the documented actor/critic/action contracts and other task defaults."""
    from isaaclab_tasks.contrib.forge.forge_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ForgeTaskPegInsertCfg
    from isaaclab_tasks.contrib.forge.forge_tasks_cfg import ForgeGearMesh, ForgeNutThread

    cfg = ForgeTaskPegInsertCfg()
    assert sum(OBS_DIM_CFG[key] for key in cfg.obs_order) + cfg.action_space == 24
    assert sum(STATE_DIM_CFG[key] for key in cfg.state_order) + cfg.action_space == 61
    assert cfg.action_space == 7
    assert cfg.task.z_axis_align_scale == -1.0
    assert cfg.task.z_axis_align_alpha_range == [10.0, 20.0]
    assert cfg.task.z_axis_align_kappa == 0.1756542881
    assert ForgeGearMesh().z_axis_align_scale == ForgeNutThread().z_axis_align_scale == 0.0
