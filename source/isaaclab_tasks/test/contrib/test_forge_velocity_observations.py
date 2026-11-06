# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled-state regression tests for FORGE velocity observations."""

from math import cos, sin
from types import SimpleNamespace

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import pytest
import torch
import warp as wp

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_euler_xyz, quat_mul

from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv


@pytest.fixture
def env():
    """Exercise both production compute methods using controlled raw sensor tensors."""
    instance = object.__new__(ForgeEnv)
    instance._is_closed = True
    instance.sim = SimpleNamespace(device="cpu")
    instance.scene = SimpleNamespace(num_envs=2, env_origins=torch.zeros(2, 3))
    instance.cfg_task = SimpleNamespace(name="peg_insert")
    instance.cfg = SimpleNamespace(
        obs_rand=SimpleNamespace(fingertip_pos=0.0, fingertip_rot_deg=0.0, ft_force=0.0),
        ft_smoothing_factor=0.25,
    )
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    root_data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        root_quat_w=SimpleNamespace(torch=identity.clone()),
    )
    instance._fixed_asset = SimpleNamespace(data=root_data)
    instance._held_asset = SimpleNamespace(data=root_data)
    data = SimpleNamespace(
        body_pos_w=SimpleNamespace(torch=torch.zeros(2, 3, 3)),
        body_quat_w=SimpleNamespace(torch=identity[:, None, :].repeat(1, 3, 1)),
        body_lin_vel_w=SimpleNamespace(torch=torch.full((2, 3, 3), 0.2)),
        body_ang_vel_w=SimpleNamespace(torch=torch.full((2, 3, 3), 0.3)),
        body_link_jacobian_w=SimpleNamespace(torch=torch.zeros(2, 2, 6, 9)),
        mass_matrix=SimpleNamespace(torch=torch.eye(9).repeat(2, 1, 1)),
        joint_pos=SimpleNamespace(torch=torch.zeros(2, 9)),
        joint_vel=SimpleNamespace(torch=torch.zeros(2, 9)),
        _sim_timestamp=0.0,
    )
    force = wp.from_torch(torch.zeros(2, 1, 6))
    instance._robot = SimpleNamespace(
        data=data, _data=data, root_view=SimpleNamespace(get_link_incoming_joint_force=lambda: force)
    )
    instance.fingertip_body_idx = 0
    instance.left_finger_body_idx = 1
    instance.right_finger_body_idx = 2
    instance.force_sensor_body_idx = 0
    instance.force_sensor_world_smooth = torch.zeros(2, 6)
    instance.fixed_pos_obs_frame = torch.zeros(2, 3)
    instance.init_fixed_pos_obs_noise = torch.zeros(2, 3)
    instance.flip_quats = torch.ones(2)
    instance.prev_fingertip_pos = torch.zeros(2, 3)
    instance.prev_fingertip_quat = identity.clone()
    instance.prev_joint_pos = torch.zeros(2, 7)
    instance._prev_noisy_fingertip_pos = None
    instance._prev_noisy_fingertip_quat = None
    return instance


def _sample(env, position, yaw=0.0, dt=1 / 120):
    env._robot.data.body_pos_w.torch[:, 0] = torch.as_tensor(position)
    env._robot.data.body_quat_w.torch[:, 0] = quat_from_euler_xyz(
        torch.full((2,), torch.pi), torch.zeros(2), torch.full((2,), float(yaw))
    )
    env._robot._data._sim_timestamp += dt
    env._compute_intermediate_values(dt)


def test_first_sample_has_zero_velocity(env):
    _sample(env, [0.5, -0.2, 0.1], yaw=0.7)
    torch.testing.assert_close(env.ee_linvel_fd, torch.zeros(2, 3))
    torch.testing.assert_close(env.ee_angvel_fd, torch.zeros(2, 3))


@pytest.mark.parametrize("dt", [1 / 120, 1 / 60])
def test_translation_uses_previous_sample(env, dt):
    start = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.1]])
    velocity = torch.tensor([[0.12, -0.24, 0.36], [-0.36, 0.12, 0.0]])
    _sample(env, start, dt=dt)
    _sample(env, start + velocity * dt, dt=dt)
    torch.testing.assert_close(env.ee_linvel_fd, velocity, atol=3e-6, rtol=1e-5)
    # The parent history and controller velocity retain the raw simulator state.
    torch.testing.assert_close(env.prev_fingertip_pos, env.fingertip_midpoint_pos)
    torch.testing.assert_close(env.fingertip_midpoint_linvel, torch.full((2, 3), 0.2))


def test_stationary_samples_have_zero_velocity(env):
    _sample(env, [0.1, 0.2, 0.3], yaw=0.7)
    _sample(env, [0.1, 0.2, 0.3], yaw=0.7)
    torch.testing.assert_close(env.ee_linvel_fd, torch.zeros(2, 3))
    tolerance = 2 * torch.finfo(torch.float32).eps / (1 / 120)
    torch.testing.assert_close(env.ee_angvel_fd, torch.zeros(2, 3), atol=tolerance, rtol=0)


@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_yaw_motion_survives_projection_and_quaternion_sign(env, flip):
    _sample(env, [0.0, 0.0, 0.0], yaw=0.7)
    env.flip_quats[:] = flip
    yaw_step = 0.01
    _sample(env, [0.0, 0.0, 0.0], yaw=0.7 + yaw_step)
    expected = torch.zeros(2, 3)
    expected[:, 2] = yaw_step / (1 / 120)
    torch.testing.assert_close(env.ee_angvel_fd, expected, atol=2e-5, rtol=1e-5)
    assert torch.count_nonzero(env.noisy_fingertip_quat[:, [2, 3]]) == 0


@pytest.mark.parametrize("task_name", ["peg_insert", "gear_mesh", "nut_thread"])
@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_observation_projection_preserves_selected_components(env, task_name, flip):
    env.cfg_task.name = task_name
    env.flip_quats[:] = flip
    full_quat = quat_from_euler_xyz(
        torch.tensor([torch.pi - 0.2, torch.pi]), torch.tensor([0.3, 0.0]), torch.tensor([0.7, 0.0])
    )
    env._robot.data.body_quat_w.torch[:, 0] = full_quat
    env._compute_intermediate_values(1 / 120)

    expected = full_quat * flip
    if task_name == "peg_insert":
        expected[:, 2:] = 0.0
    else:
        expected[:, [0, 3]] = 0.0
    torch.testing.assert_close(env.noisy_fingertip_quat, expected, atol=1e-7, rtol=1e-6)


def test_velocity_includes_both_position_noise_samples(env, monkeypatch):
    env.cfg.obs_rand.fingertip_pos = 0.00025
    noises = [torch.full((2, 3), 1.0), torch.full((2, 3), -0.5)]
    original_randn = torch.randn
    sample_index = 0

    def controlled_randn(*args, **kwargs):
        nonlocal sample_index
        # Each production sample draws position, rotation axis, angle, then force noise.
        value = noises[sample_index // 4] if sample_index % 4 == 0 else original_randn(*args, **kwargs)
        sample_index += 1
        return value

    monkeypatch.setattr(torch, "randn", controlled_randn)
    _sample(env, [0.0, 0.0, 0.0])
    _sample(env, [0.001, 0.0, 0.0])
    expected = (torch.tensor([0.001, 0.0, 0.0]) + (noises[1] - noises[0]) * 0.00025) / (1 / 120)
    torch.testing.assert_close(env.ee_linvel_fd, expected)
    assert sample_index == 8


def test_velocity_includes_both_rotation_noise_samples(env, monkeypatch):
    env.cfg.obs_rand.fingertip_rot_deg = 1.0
    angles = [0.4, -0.2]
    sample_index = 0

    def controlled_randn(shape, **kwargs):
        nonlocal sample_index
        value = torch.zeros(shape, **kwargs)
        if sample_index % 4 == 1:
            value[:, 2] = 1.0
        elif sample_index % 4 == 2:
            value[:] = angles[sample_index // 4]
        sample_index += 1
        return value

    monkeypatch.setattr(torch, "randn", controlled_randn)
    _sample(env, [0.0, 0.0, 0.0], yaw=0.7)
    _sample(env, [0.0, 0.0, 0.0], yaw=0.7)
    expected = torch.zeros(2, 3)
    # Local z points downward at roll=pi, so positive sensor noise gives negative world yaw.
    expected[:, 2] = -(angles[1] - angles[0]) * torch.pi / 180 / (1 / 120)
    torch.testing.assert_close(env.ee_angvel_fd, expected, atol=3e-5, rtol=1e-5)
    noisy_yaw = 0.7 - angles[1] * torch.pi / 180
    expected_quat = torch.tensor([cos(noisy_yaw / 2), sin(noisy_yaw / 2), 0.0, 0.0])
    torch.testing.assert_close(env.noisy_fingertip_quat, expected_quat.repeat(2, 1), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("task_name", ["gear_mesh", "nut_thread"])
def test_other_forge_tasks_keep_legacy_velocity(env, task_name):
    env.cfg_task.name = task_name
    env.cfg.obs_rand.fingertip_pos = 0.00025
    for yaw in (0.7, 0.8):
        _sample(env, [0.1, 0.2, 0.3], yaw=yaw)
        expected_linvel = (env.noisy_fingertip_pos - env.fingertip_midpoint_pos) / (1 / 120)
        relative = quat_mul(env.noisy_fingertip_quat, quat_conjugate(env.fingertip_midpoint_quat))
        relative *= torch.sign(relative[:, 3:4])
        expected_angvel = axis_angle_from_quat(relative) / (1 / 120)
        expected_angvel[:, :2] = 0.0
        torch.testing.assert_close(env.ee_linvel_fd, expected_linvel)
        torch.testing.assert_close(env.ee_angvel_fd, expected_angvel)
