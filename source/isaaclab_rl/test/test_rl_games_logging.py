# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for RL-Games training episode namespaces."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch

from isaaclab_rl.rl_games._algo_observer import _IsaacLabAlgoObserver


def test_alignment_reports_are_sparse_averaged_and_keep_their_namespace() -> None:
    """Only completed alignment reports contribute; existing training tags stay unchanged."""
    scalars: list[tuple[str, float, int | float]] = []
    flushes: list[bool] = []
    writer = SimpleNamespace(
        add_scalar=lambda name, value, step: scalars.append((name, float(value), step)),
        flush=lambda: flushes.append(True),
    )
    observer = _IsaacLabAlgoObserver()
    observer.after_init(SimpleNamespace(writer=writer, device="cpu", ppo_device="cpu", games_to_track=10))
    for episode in (
        {"F_mean": 3.0},
        {"F_mean": 3.0, "ZAxisAlign/z_axis_angle_mean_deg": 2.0},
        {"F_mean": 3.0},
        {"F_mean": 3.0, "ZAxisAlign/z_axis_angle_mean_deg": 8.0},
    ):
        observer.process_infos(
            {
                "episode": episode,
                "logs_rew_z_axis_align": torch.tensor(0.75),
                "logs_rew_z_axis_align_weighted": torch.tensor(0.075),
            },
            torch.tensor([0]),
        )
    observer.after_print_stats(frame=100, epoch_num=3, total_time=7.0)
    assert ("ZAxisAlign/z_axis_angle_mean_deg", 5.0, 3) in scalars
    assert ("Episode/F_mean", 3.0, 3) in scalars
    for suffix, step in (("frame", 100), ("iter", 3), ("time", 7.0)):
        assert (f"logs_rew_z_axis_align/{suffix}", 0.75, step) in scalars
        assert (f"logs_rew_z_axis_align_weighted/{suffix}", torch.tensor(0.075).item(), step) in scalars
    assert not any(name.startswith("Episode/ZAxisAlign/") for name, _, _ in scalars)
    assert len(flushes) == 1
    scalars.clear()
    observer.after_print_stats(frame=200, epoch_num=4, total_time=9.0)
    assert not any(name.startswith("ZAxisAlign/") for name, _, _ in scalars)
    assert len(flushes) == 1


def test_alignment_metrics_pass_through_wrapper_and_evaluator(tmp_path: Path) -> None:
    """The real wrapper retains episode metadata and the evaluator preserves all nine tags."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import DirectRLEnv

    from isaaclab_rl.entrypoints.backends.evaluate_rl_games import EpisodeAccumulator, _write_report
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    names = [
        "z_axis_align_reward_mean",
        "z_axis_align_weighted_mean",
        "z_axis_align_weighted_return",
        "z_axis_angle_mean_deg",
        "z_axis_angle_p95_deg",
        "z_axis_angle_final_deg",
        "z_axis_aligned_fraction_5deg",
        "z_axis_align_reference_5deg_mean",
        "z_axis_align_sigma_deg",
    ]
    episode_metrics = {f"ZAxisAlign/{name}": float(index) for index, name in enumerate(names)}

    class FakeDirectEnv(DirectRLEnv):
        def __init__(self):
            self._is_closed = True
            self.cfg = SimpleNamespace(is_finite_horizon=True)
            self.scene = SimpleNamespace(num_envs=2)
            self.sim = SimpleNamespace(device="cpu")
            self.single_observation_space = {
                "policy": gym.spaces.Box(-1, 1, (3,)),
                "critic": gym.spaces.Box(-1, 1, (4,)),
            }
            self.single_action_space = gym.spaces.Box(-1, 1, (7,))

        def step(self, actions):
            assert actions.shape == (2, 7)
            return (
                {"policy": torch.zeros((2, 3)), "critic": torch.zeros((2, 4))},
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, dtype=torch.bool),
                {"log": {**episode_metrics, "F_mean": 3.0}, "logs_rew_z_axis_align": torch.tensor(0.75)},
            )

    wrapper = RlGamesVecEnvWrapper(FakeDirectEnv(), "cpu", 10.0, 1.0)
    observations, rewards, dones, infos = wrapper.step(torch.zeros((2, 7)))
    assert observations["obs"].shape == (2, 3)
    assert observations["states"].shape == (2, 4)
    assert "log" not in infos
    accumulator = EpisodeAccumulator(2, 1)
    accumulator.update(rewards, dones, infos)
    report = accumulator.report(iteration=1, elapsed_time=1.0)
    for name, value in episode_metrics.items():
        assert report[name] == value
        assert f"Episode/{name}" not in report
    assert report["Episode/F_mean"] == 3.0
    with SummaryWriter(str(tmp_path)) as writer:
        _write_report(writer, None, report, accumulator.frames, iteration=1, elapsed_time=3.0)
    events = EventAccumulator(str(tmp_path)).Reload()
    for name, value in episode_metrics.items():
        event = events.Scalars(name)[0]
        assert event.value == value
        assert event.step == 1
    assert events.Scalars("logs_rew_z_axis_align/frame")[0].step == 2
    assert events.Scalars("logs_rew_z_axis_align/iter")[0].step == 1
    assert events.Scalars("logs_rew_z_axis_align/time")[0].step == 3
