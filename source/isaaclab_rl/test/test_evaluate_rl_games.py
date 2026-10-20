# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for inference-only RL-Games checkpoint evaluation helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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
                    "ZAxisAlign/z_axis_angle_mean_deg": torch.tensor(4.0),
                },
                "logs_rew_contact_penalty": torch.tensor(0.25),
                "logs_rew_z_axis_align": torch.tensor(0.8),
                "logs_rew_z_axis_align_weighted": torch.tensor(0.08),
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
    assert summary["ZAxisAlign/z_axis_angle_mean_deg"] == 4.0
    assert "Episode/ZAxisAlign/z_axis_angle_mean_deg" not in summary
    for suffix in ("frame", "iter", "time"):
        assert summary[f"logs_rew_z_axis_align/{suffix}"] == torch.tensor(0.8).item()
        assert summary[f"logs_rew_z_axis_align_weighted/{suffix}"] == torch.tensor(0.08).item()
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
            "episode": {
                "Metrics/success_rate": 1.0,
                "F_mean": 3.5,
                "F_max": 7.0,
                "ZAxisAlign/z_axis_angle_mean_deg": 4.0,
            },
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
    assert ("ZAxisAlign/z_axis_angle_mean_deg", 4.0, 1) in writer.scalars
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
