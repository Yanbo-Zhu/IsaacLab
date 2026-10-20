# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent contracts for FORGE z-axis rewards and episode statistics."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from isaaclab_tasks.contrib.forge.forge_utils import (
    _compute_z_axis_alignment,
    _EpisodeZAxisAlignmentStatistics,
    _ZAxisAlignmentReward,
)


def _quaternions(roll: list[float], pitch: list[float], yaw: list[float]) -> torch.Tensor:
    return quat_from_euler_xyz(
        *(torch.deg2rad(torch.tensor(angles, dtype=torch.float64)) for angles in (roll, pitch, yaw))
    )


def test_z_axis_reward_matches_an_independent_angle_reference() -> None:
    """The reward is directional, monotone, and has the specified angular scale."""
    angles = [0.0, 2.0, 5.0, 10.0, 90.0, 180.0]
    held = _quaternions(angles, [0.0] * 6, [0.0] * 6)
    fixed = _quaternions([0.0] * 6, [0.0] * 6, [0.0] * 6)
    sigma = torch.full((6,), math.radians(5.0), dtype=torch.float64)
    reward, angle_deg, reference = _compute_z_axis_alignment(held, fixed, sigma)
    expected = torch.tensor(
        [math.exp(-(1 - math.cos(math.radians(angle))) / (1 - math.cos(math.radians(5)))) for angle in angles],
        dtype=torch.float64,
    )
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(reference, expected)
    torch.testing.assert_close(angle_deg, torch.tensor(angles, dtype=torch.float64))
    assert torch.all(reward[:-1] > reward[1:])
    assert reward[2].item() == pytest.approx(math.exp(-1))


def test_z_axis_reward_is_invariant_to_local_spin_sign_and_common_rotation() -> None:
    """Local z spin, quaternion sign, and reference-frame changes preserve alignment."""
    held = _quaternions([13.0], [7.0], [24.0])
    fixed = _quaternions([-4.0], [11.0], [-33.0])
    sigma = torch.tensor([math.radians(5.0)], dtype=torch.float64)
    expected = _compute_z_axis_alignment(held, fixed, sigma)
    local_spin = _quaternions([0.0], [0.0], [71.0])
    common_rotation = _quaternions([31.0], [-17.0], [53.0])
    for held_test, fixed_test in (
        (-held, fixed),
        (held, -fixed),
        (quat_mul(held, local_spin), fixed),
        (held, quat_mul(fixed, local_spin)),
        (quat_mul(common_rotation, held), quat_mul(common_rotation, fixed)),
    ):
        for actual, target in zip(_compute_z_axis_alignment(held_test, fixed_test, sigma), expected):
            torch.testing.assert_close(actual, target)


@pytest.mark.parametrize(
    "scale,sigma_range",
    [
        (-1.0, [2.0, 10.0]),
        (math.nan, [2.0, 10.0]),
        (math.inf, [2.0, 10.0]),
        (0.1, [5.0]),
        (0.1, [0.0, 10.0]),
        (0.1, [10.0, 2.0]),
        (0.1, [2.0, 181.0]),
        (0.1, [math.nan, 10.0]),
        (0.1, [2.0, math.inf]),
        (0.1, [1e-30, 1e-30]),
    ],
)
def test_z_axis_reward_rejects_invalid_configuration(scale: float, sigma_range: list[float]) -> None:
    with pytest.raises(ValueError):
        _ZAxisAlignmentReward(2, "cpu", 7, scale, sigma_range)


def test_sigma_sampling_is_seeded_isolated_and_limited_to_reset_environments() -> None:
    """Only reset environments resample; sampling never advances the global RNG."""
    first = _ZAxisAlignmentReward(3, "cpu", 7, 0.1, [2.0, 10.0])
    second = _ZAxisAlignmentReward(3, "cpu", 7, 0.1, [2.0, 10.0])
    global_state = torch.random.get_rng_state().clone()
    for ids in (torch.arange(3), torch.tensor([0, 2])):
        unchanged = first.sigma[1].clone()
        first.reset(ids)
        second.reset(ids)
        torch.testing.assert_close(first.sigma, second.sigma)
        if ids.numel() == 2:
            assert first.sigma[1] == unchanged
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.all((first.sigma >= math.radians(2)) & (first.sigma <= math.radians(10)))
    before = first.sigma.clone()
    identity = _quaternions([0.0] * 3, [0.0] * 3, [0.0] * 3).float()
    first.compute(identity, identity)
    torch.testing.assert_close(first.sigma, before)


def test_fixed_sigma_does_not_consume_random_numbers() -> None:
    reward = _ZAxisAlignmentReward(2, "cpu", 7, 0.0, [5.0, 5.0])
    state = reward.generator.get_state().clone()
    reward.reset(torch.arange(2))
    torch.testing.assert_close(reward.sigma, torch.full((2,), math.radians(5)))
    assert torch.equal(reward.generator.get_state(), state)


def test_episode_statistics_include_final_sample_and_isolate_partial_resets() -> None:
    """Episode reductions use each trajectory's valid samples, including its last step."""
    statistics = _EpisodeZAxisAlignmentStatistics(2, 4, "cpu")
    statistics.update(torch.tensor([0.2, 0.4]), torch.tensor([0.0, 10.0]), torch.tensor([0.5, 0.3]))
    statistics.update(torch.tensor([0.6, 0.8]), torch.tensor([5.0, 20.0]), torch.tensor([0.7, 0.1]))
    sigma = torch.deg2rad(torch.tensor([2.0, 10.0]))
    metrics = statistics.report(torch.arange(2), sigma, 0.1)
    expected = {
        "z_axis_align_reward_mean": 0.5,
        "z_axis_align_weighted_mean": 0.05,
        "z_axis_align_weighted_return": 0.1,
        "z_axis_angle_mean_deg": 8.75,
        "z_axis_angle_p95_deg": (4.75 + 19.5) / 2,
        "z_axis_angle_final_deg": 12.5,
        "z_axis_aligned_fraction_5deg": 0.5,
        "z_axis_align_reference_5deg_mean": 0.4,
        "z_axis_align_sigma_deg": 6.0,
    }
    assert set(metrics) == {f"ZAxisAlign/{name}" for name in expected}
    for name, value in expected.items():
        assert metrics[f"ZAxisAlign/{name}"].item() == pytest.approx(value)

    statistics.reset(torch.tensor([0]))
    statistics.update(torch.tensor([1.0, 0.0]), torch.tensor([30.0, 40.0]), torch.tensor([0.0, 0.2]))
    first = statistics.report(torch.tensor([0]), sigma, 0.0)
    second = statistics.report(torch.tensor([1]), sigma, 0.1)
    assert first["ZAxisAlign/z_axis_angle_mean_deg"] == 30.0
    assert first["ZAxisAlign/z_axis_align_weighted_return"] == 0.0
    assert second["ZAxisAlign/z_axis_angle_mean_deg"].item() == pytest.approx(70.0 / 3)
    assert second["ZAxisAlign/z_axis_align_weighted_return"].item() == pytest.approx(0.12)
    # Reports own their values even after the source buffers are reset.
    assert metrics["ZAxisAlign/z_axis_angle_final_deg"] == 12.5


@pytest.mark.parametrize("play_mode", [False, True])
def test_alignment_hydra_settings_serialize_without_changing_network_inputs(play_mode: bool) -> None:
    """Training and play/evaluation resolution retain the two new task fields."""
    from isaaclab_tasks.contrib.forge.forge_env_cfg import ForgeTaskPegInsertCfg
    from isaaclab_tasks.contrib.forge.forge_tasks_cfg import ForgeGearMesh, ForgeNutThread
    from isaaclab_tasks.utils import resolve_task_config

    default = ForgeTaskPegInsertCfg()
    assert default.task.z_axis_align_scale == 0.1
    assert default.task.z_axis_align_sigma_range == [2.0, 10.0]
    assert ForgeGearMesh().z_axis_align_scale == ForgeNutThread().z_axis_align_scale == 0.0
    cfg, _ = resolve_task_config(
        "IsaacContrib-Forge-PegInsert-Direct",
        "rl_games_cfg_entry_point",
        play_mode=play_mode,
        overrides=["env.task.z_axis_align_scale=0.0", "env.task.z_axis_align_sigma_range=[5.0,5.0]"],
    )
    assert cfg.to_dict()["task"]["z_axis_align_scale"] == 0.0
    assert cfg.to_dict()["task"]["z_axis_align_sigma_range"] == [5.0, 5.0]
    assert cfg.action_space == default.action_space == 7
    assert cfg.obs_order == default.obs_order
    assert cfg.state_order == default.state_order


def test_forge_reward_preserves_existing_terms_and_completed_reports() -> None:
    """Exercise the actual reward methods with CPU states and no physics stepping."""
    pytest.importorskip("carb", exc_type=ImportError)
    from isaaclab_tasks.contrib.factory.factory_env import FactoryEnv
    from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv
    from isaaclab_tasks.contrib.forge.forge_env_cfg import ForgeTaskPegInsertCfg
    from isaaclab_tasks.contrib.forge.forge_utils import _EpisodeForceStatistics

    env = ForgeEnv.__new__(ForgeEnv)
    env._is_closed = True
    env.cfg = ForgeTaskPegInsertCfg()
    env.cfg_task = env.cfg.task
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device="cpu")
    env.extras = {}
    env.held_pos = torch.tensor([[0.0, 0.0, 0.03], [0.0, 0.0, 0.03]])
    env.fixed_pos = torch.zeros((2, 3))
    env.held_quat = _quaternions([0.0, 5.0], [0.0, 0.0], [0.0, 0.0]).float()
    env.fixed_quat = _quaternions([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]).float()
    env.actions = env.prev_actions = torch.zeros((2, 7))
    env.delta_pos = torch.zeros((2, 3))
    env.delta_yaw = torch.zeros(2)
    env.force_sensor_smooth = torch.tensor([[7.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    env.contact_penalty_thresholds = torch.full((2,), 5.0)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.ep_succeeded = torch.zeros(2, dtype=torch.long)
    env.ep_success_times = torch.zeros(2, dtype=torch.long)
    env.first_pred_success_tx = {}
    env.success_pred_scale = 0.0
    env._episode_force_statistics = _EpisodeForceStatistics(2, "cpu")
    env._z_axis_alignment = _ZAxisAlignmentReward(2, "cpu", 7, 0.0, [5.0, 5.0])
    env._episode_z_axis_statistics = _EpisodeZAxisAlignmentStatistics(2, 4, "cpu")

    original = FactoryEnv._get_rewards(env) - torch.tensor([2 * env.cfg_task.contact_penalty_scale, 0.0])
    torch.testing.assert_close(env._get_rewards(), original, rtol=0, atol=0)
    assert not any(name.startswith("ZAxisAlign/") for name in env.extras["log"])

    env._z_axis_alignment.scale = 0.1
    env.reset_buf[:] = True
    env.episode_length_buf[:] = 2
    expected_reward, _, _ = env._z_axis_alignment.compute(env.held_quat, env.fixed_quat)
    torch.testing.assert_close(env._get_rewards(), original + 0.1 * expected_reward)
    completed_report = env.extras["log"]
    assert len([name for name in completed_report if name.startswith("ZAxisAlign/")]) == 9
    assert completed_report["ZAxisAlign/z_axis_angle_final_deg"].item() == pytest.approx(2.5, abs=1e-4)

    env._episode_z_axis_statistics.reset(torch.arange(2))
    env.reset_buf[:] = False
    env._get_rewards()
    assert not any(name.startswith("ZAxisAlign/") for name in env.extras["log"])
    assert completed_report["ZAxisAlign/z_axis_angle_final_deg"].item() == pytest.approx(2.5, abs=1e-4)
