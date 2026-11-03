# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for inference-only RL-Games checkpoint evaluation helpers."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from isaaclab_rl.entrypoints.backends.evaluate_rl_games import (
    EpisodeAccumulator,
    _load_player,
    _rollout,
    _start_wandb_run,
    _write_report,
)


class _FakeEnv:
    """Minimal vectorized environment with two synchronized two-step episodes."""

    def __init__(self) -> None:
        self.unwrapped = SimpleNamespace(num_envs=2)
        self._step = 0

    def reset(self) -> torch.Tensor:
        return torch.zeros((2, 1))

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        del actions
        self._step += 1
        dones = torch.tensor([self._step % 2 == 0, self._step % 2 == 0])
        infos: dict[str, object] = {}
        if bool(torch.any(dones)):
            infos = {
                "episode": {
                    "Metrics/success_rate": torch.tensor(0.5),
                    "F_mean": torch.tensor(3.5),
                    "F_max": torch.tensor(7.0),
                },
                "logs_rew_contact_penalty": torch.tensor(0.25),
                "early_term_precision/0.5": 0.75,
            }
        return torch.zeros((2, 1)), torch.tensor([1.0, 2.0]), dones, infos


class _FakePlayer:
    """Player that exposes the RL-Games methods used by the evaluator."""

    def __init__(self) -> None:
        self.is_rnn = True
        self.is_deterministic = True
        self.states = [torch.ones((1, 2, 1))]

    def get_batch_size(self, observations: torch.Tensor, batch_size: int) -> int:
        del observations
        return batch_size

    def init_rnn(self) -> None:
        self.states = [torch.ones((1, 2, 1))]

    def obs_to_torch(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

    def get_action(self, observations: torch.Tensor, *, is_deterministic: bool) -> torch.Tensor:
        assert is_deterministic
        return torch.zeros_like(observations)


def test_rollout_collects_fixed_episodes_and_training_compatible_metrics() -> None:
    """Evaluation ends after the requested count per environment and preserves metric names."""
    player = _FakePlayer()
    reports: list[dict[str, float]] = []

    accumulator = _rollout(
        _FakeEnv(),
        player,
        num_episodes=2,
        on_completed_episodes=lambda value: reports.append(value.report(len(reports) + 1, elapsed_time=2.0)),
    )
    summary = accumulator.summary(iteration=len(reports), elapsed_time=2.0)

    assert summary["evaluation/episodes"] == 4.0
    assert summary["rewards/step"] == 3.0
    assert summary["rewards/iter"] == 3.0
    assert summary["rewards/time"] == 3.0
    assert summary["episode_lengths/step"] == 2.0
    assert summary["episode_lengths/iter"] == 2.0
    assert summary["episode_lengths/time"] == 2.0
    assert summary["Episode/Metrics/success_rate"] == 0.5
    assert summary["Episode/F_mean"] == 3.5
    assert summary["Episode/F_max"] == 7.0
    assert summary["logs_rew_contact_penalty/frame"] == 0.25
    assert summary["logs_rew_contact_penalty/iter"] == 0.25
    assert summary["logs_rew_contact_penalty/time"] == 0.25
    assert summary["early_term_precision/0.5/frame"] == 0.75
    assert len(reports) == 2
    assert reports[0]["evaluation/episodes"] == 2.0
    assert reports[1]["evaluation/episodes"] == 4.0
    assert accumulator.frames == 8
    assert torch.count_nonzero(player.states[0]) == 0


def test_accumulator_rejects_summary_before_all_envs_complete() -> None:
    """Partial vectorized evaluation must not be reported as a final result."""
    accumulator = EpisodeAccumulator(num_envs=2, num_episodes=1)
    accumulator.update(torch.tensor([1.0, 1.0]), torch.tensor([True, False]), {})

    assert not accumulator.is_complete
    try:
        accumulator.summary(iteration=1, elapsed_time=1.0)
    except RuntimeError as error:
        assert "before every environment" in str(error)
    else:
        raise AssertionError("Expected a partial evaluation summary to fail.")


def test_load_player_never_calls_the_training_runner() -> None:
    """Checkpoint evaluation constructs only a player and has no route to Runner.run()."""

    class FakePlayer:
        def __init__(self) -> None:
            self.restored_checkpoint: str | None = None
            self.reset_calls = 0

        def restore(self, checkpoint: str) -> None:
            self.restored_checkpoint = checkpoint

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeRunner:
        def __init__(self) -> None:
            self.player = FakePlayer()
            self.loaded_config: dict[str, object] | None = None

        def load(self, config: dict[str, object]) -> None:
            self.loaded_config = config

        def create_player(self) -> FakePlayer:
            return self.player

        def run(self, *_args, **_kwargs) -> None:
            raise AssertionError("Evaluation must not call Runner.run().")

    config: dict[str, object] = {"params": {"load_checkpoint": False, "load_path": "", "config": {"num_actors": 0}}}
    player = _load_player(FakeRunner, config, "/tmp/policy.pth", num_actors=2)

    assert player.restored_checkpoint == "/tmp/policy.pth"
    assert player.reset_calls == 1
    assert config["params"]["load_checkpoint"] is True
    assert config["params"]["load_path"] == "/tmp/policy.pth"
    assert config["params"]["config"]["num_actors"] == 2


def test_report_writes_tensorboard_and_wandb_history() -> None:
    """Completed evaluation batches write training-compatible tags without training-only metrics."""
    accumulator = EpisodeAccumulator(num_envs=1, num_episodes=1)
    accumulator.update(
        torch.tensor([2.0]),
        torch.tensor([True]),
        {
            "episode": {"Metrics/success_rate": 1.0, "F_mean": 3.5, "F_max": 7.0},
            "logs_rew_contact_penalty": 0.25,
            "early_term_precision/0.5": torch.tensor(0.75),
        },
    )
    metrics = accumulator.report(iteration=1, elapsed_time=2.5)

    class FakeWriter:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []
            self.flush_calls = 0

        def add_scalar(self, name: str, value: float, step: int) -> None:
            self.scalars.append((name, value, step))

        def flush(self) -> None:
            self.flush_calls += 1

    class FakeWandbRun:
        def __init__(self) -> None:
            self.history: list[tuple[dict[str, float | int], int]] = []

        def log(self, values: dict[str, float | int], *, step: int) -> None:
            self.history.append((values, step))

    writer = FakeWriter()
    wandb_run = FakeWandbRun()
    _write_report(writer, wandb_run, metrics, accumulator.frames, iteration=1, elapsed_time=2.5)

    assert ("Episode/Metrics/success_rate", 1.0, 1) in writer.scalars
    assert ("Episode/F_mean", 3.5, 1) in writer.scalars
    assert ("Episode/F_max", 7.0, 1) in writer.scalars
    assert ("logs_rew_contact_penalty/frame", 0.25, 1) in writer.scalars
    assert ("logs_rew_contact_penalty/iter", 0.25, 1) in writer.scalars
    assert ("logs_rew_contact_penalty/time", 0.25, 2) in writer.scalars
    assert ("rewards/step", 2.0, 1) in writer.scalars
    assert ("rewards/iter", 2.0, 1) in writer.scalars
    assert ("rewards/time", 2.0, 2) in writer.scalars
    assert writer.flush_calls == 1
    assert wandb_run.history == [({**metrics, "global_step": 1}, 1)]
    assert not any(name.startswith(("losses/", "info/", "shaped_rewards/")) for name in metrics)


def test_wandb_tracking_uses_explicit_history_reporting(monkeypatch) -> None:
    """Tracked evaluation keeps a local TensorBoard event file and sends metrics through W&B history."""
    config_updates: list[dict[str, object]] = []
    init_kwargs: dict[str, object] = {}
    run = SimpleNamespace(
        resumed=False,
        config=SimpleNamespace(update=lambda value: config_updates.append(value)),
    )

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    args = SimpleNamespace(
        track=True,
        wandb_entity="test-entity",
        wandb_project_name="test-project",
        wandb_name="evaluation-name",
        num_episodes=2,
    )
    env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), seed=0, to_dict=lambda: {"episode_length_s": 10.0})
    agent_cfg = {"params": {"config": {"name": "Forge"}}}

    result = _start_wandb_run(args, "/tmp/evaluation", env_cfg, agent_cfg, "/tmp/Forge.pth")

    assert result is run
    assert init_kwargs["project"] == "test-project"
    assert init_kwargs["entity"] == "test-entity"
    assert init_kwargs["name"] == "evaluation-name"
    assert init_kwargs["sync_tensorboard"] is False
    assert init_kwargs["monitor_gym"] is True
    assert init_kwargs["save_code"] is True
    assert config_updates == [
        {
            "env_cfg": {"episode_length_s": 10.0},
            "agent_cfg": agent_cfg,
            "evaluation": {
                "checkpoint": "/tmp/Forge.pth",
                "num_episodes_per_env": 2,
                "num_envs": 1,
                "seed": 0,
            },
        }
    ]


def _alignment_payload(ids: list[int], values: list[float]) -> dict[str, object]:
    return {"env_ids": torch.tensor(ids), "metrics": {"z_axis_angle_mean_deg": torch.tensor(values)}}


def test_alignment_evaluation_weights_episodes_and_excludes_exhausted_envs() -> None:
    """Uneven completion batches and over-quota environments retain the correct episodes."""
    accumulator = EpisodeAccumulator(num_envs=3, num_episodes=1)
    accumulator.update(
        torch.ones(3),
        torch.tensor([True, True, False]),
        {"z_axis_align": _alignment_payload([0, 1], [10.0, 20.0])},
    )
    accumulator.update(
        torch.ones(3),
        torch.tensor([True, False, True]),
        {"z_axis_align": _alignment_payload([0, 2], [1000.0, 60.0])},
    )
    report = accumulator.summary(iteration=2, elapsed_time=1.0)
    assert report["ZAxisAlign/z_axis_angle_mean_deg"] == 30.0
    assert report["evaluation/episodes"] == 3
    assert not any(name.startswith("Episode/ZAxisAlign/") for name in report)


def test_alignment_observer_keeps_legacy_metrics_and_flushes_window() -> None:
    """The real observer uses episode weighting, epoch steps, one prefix and a fresh window."""
    from isaaclab_rl.rl_games import RlGamesAlgoObserver

    scalars = []
    flushes = []
    writer = SimpleNamespace(add_scalar=lambda *value: scalars.append(value), flush=lambda: flushes.append(True))
    algo = SimpleNamespace(games_to_track=10, ppo_device="cpu", device="cpu", writer=writer)
    observer = RlGamesAlgoObserver()
    observer.after_init(algo)
    observer.process_infos({"z_axis_align": _alignment_payload([0, 1], [10, 20]), "episode": {"F_mean": 2.0}}, None)
    observer.process_infos(
        {"z_axis_align": _alignment_payload([2], [60]), "logs_rew_z_axis_align_weighted": -0.5}, None
    )
    observer.after_print_stats(100, 3, 2.0)
    assert ("ZAxisAlign/z_axis_angle_mean_deg", 30.0, 3) in scalars
    assert ("Episode/F_mean", 2.0, 3) in scalars
    assert ("logs_rew_z_axis_align_weighted/frame", -0.5, 100) in scalars
    assert len(flushes) == 1
    scalars.clear()
    observer.after_print_stats(200, 4, 3.0)
    assert not any(name.startswith("ZAxisAlign/") for name, *_ in scalars)
    observer.process_infos({"z_axis_align": _alignment_payload([0], [90])}, None)
    observer.after_print_stats(300, 5, 4.0)
    assert ("ZAxisAlign/z_axis_angle_mean_deg", 90.0, 5) in scalars


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_alignment_all_metrics_reach_writer_with_report_step(device: str) -> None:
    """Nested metrics may remain on sim device while terminal flags live on the RL device."""
    from isaaclab_rl.rl_games.metrics import _ZAxisAlignMetrics

    from isaaclab_tasks.contrib.forge.forge_utils import _EpisodeZAxisAlignStatistics

    stats = _EpisodeZAxisAlignStatistics(1, 2, device, -1)
    alpha = torch.tensor([0.2], device=device)
    stats.update(
        torch.tensor([0.5], device=device),
        torch.tensor([0.1], device=device),
        torch.tensor([0.2], device=device),
        alpha,
    )
    payload = {
        "env_ids": torch.tensor([0], device=device),
        "metrics": stats.report(torch.tensor([0], device=device), alpha),
    }
    accumulator = EpisodeAccumulator(1, 1)
    accumulator.update(torch.ones(1), torch.ones(1, dtype=torch.bool), {"z_axis_align": payload})
    scalars = []
    writer = SimpleNamespace(add_scalar=lambda *value: scalars.append(value), flush=lambda: None)
    metrics = accumulator.report(3, 2.0)
    _write_report(writer, None, metrics, 100, 3, 2.0)
    emitted = {name: (value, step) for name, value, step in scalars if name.startswith("ZAxisAlign/")}
    assert len(emitted) == 13
    assert all(step == 3 and math.isfinite(value) for value, step in emitted.values())
    rejected = _ZAxisAlignMetrics()
    rejected.update(payload, torch.tensor([False]))
    assert rejected.report() == {}


def test_alignment_observer_closes_real_async_writer_before_readback(tmp_path) -> None:
    """A short run must persist every final metric before an immediate Kit shutdown."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboardX import SummaryWriter

    from isaaclab_rl.rl_games import RlGamesAlgoObserver

    writer = SummaryWriter(str(tmp_path))
    observer = RlGamesAlgoObserver()
    observer.after_init(SimpleNamespace(games_to_track=10, ppo_device="cpu", device="cpu", writer=writer))
    payload = {
        "env_ids": torch.tensor([0]),
        "metrics": {f"metric_{index}": torch.tensor([float(index)]) for index in range(13)},
    }
    observer.process_infos({"z_axis_align": payload}, None)
    observer.after_print_stats(100, 2, 1.0)
    observer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert set(events.Tags()["scalars"]) == {f"ZAxisAlign/metric_{index}" for index in range(13)}
    for tag in events.Tags()["scalars"]:
        assert events.Scalars(tag)[0].step == 2
