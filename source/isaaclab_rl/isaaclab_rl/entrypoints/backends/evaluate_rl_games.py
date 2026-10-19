# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-episode, inference-only checkpoint evaluation for RL-Games."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from distutils.util import strtobool
from statistics import fmean, pstdev
from typing import Any

import torch

from isaaclab.app import add_launcher_args

from isaaclab_rl.entrypoints.common import (
    add_frontend_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    write_run_manifest,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer command-line value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


class EpisodeAccumulator:
    """Collect completed evaluation trajectories without changing the environment.

    Args:
        num_envs: Number of vectorized environments.
        num_episodes: Number of completed trajectories to retain for each environment.
    """

    def __init__(self, num_envs: int, num_episodes: int) -> None:
        self._num_episodes = num_episodes
        self._completed = torch.zeros(num_envs, dtype=torch.long)
        self._returns: torch.Tensor | None = None
        self._lengths: torch.Tensor | None = None
        self._episode_returns: list[float] = []
        self._episode_lengths: list[float] = []
        self._episode_metrics: dict[str, list[float]] = defaultdict(list)
        self._direct_metrics: dict[str, float] = {}
        self._frames = 0

    @property
    def is_complete(self) -> bool:
        """Whether every vectorized environment has supplied enough episodes."""
        return bool(torch.all(self._completed >= self._num_episodes).item())

    @property
    def completed_episodes(self) -> int:
        """Number of retained trajectories across all environments."""
        return len(self._episode_returns)

    @property
    def frames(self) -> int:
        """Number of vectorized environment transitions processed by the evaluation."""
        return self._frames

    def update(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict[str, Any]) -> bool:
        """Accumulate one vectorized environment step.

        Args:
            rewards: Per-environment rewards for the step.
            dones: Per-environment terminal flags for the step.
            infos: RL-Games-compatible step metadata.

        Returns:
            Whether this step retained at least one completed trajectory.
        """
        rewards = rewards.detach().flatten()
        dones = dones.detach().flatten().to(dtype=torch.bool)
        if rewards.numel() != self._completed.numel() or dones.numel() != self._completed.numel():
            raise ValueError("Rewards and dones must contain one value per environment.")

        if self._returns is None:
            self._returns = torch.zeros_like(rewards, dtype=torch.float)
            self._lengths = torch.zeros_like(rewards, dtype=torch.long)

        self._frames += rewards.numel()
        self._returns += rewards
        self._lengths += 1

        completed = self._completed.to(device=dones.device)
        retained_dones = torch.logical_and(dones, completed < self._num_episodes)
        if torch.any(retained_dones):
            for env_id in torch.nonzero(retained_dones, as_tuple=False).flatten().tolist():
                self._episode_returns.append(float(self._returns[env_id].item()))
                self._episode_lengths.append(float(self._lengths[env_id].item()))
                self._completed[env_id] += 1
            self._record_metrics(infos)

        if torch.any(dones):
            self._returns[dones] = 0.0
            self._lengths[dones] = 0

        return bool(torch.any(retained_dones).item())

    def report(self, iteration: int, elapsed_time: float) -> dict[str, float]:
        """Return cumulative training-compatible metrics for one evaluation report.

        Args:
            iteration: One-based evaluation report index.
            elapsed_time: Elapsed evaluation wall-clock time [s].
        """
        if not self._episode_returns:
            raise RuntimeError("Cannot report evaluation metrics before any trajectory completes.")

        mean_reward = fmean(self._episode_returns)
        mean_length = fmean(self._episode_lengths)
        metrics = {
            "rewards/step": mean_reward,
            "rewards/iter": mean_reward,
            "rewards/time": mean_reward,
            "episode_lengths/step": mean_length,
            "episode_lengths/iter": mean_length,
            "episode_lengths/time": mean_length,
            "evaluation/reward_std": pstdev(self._episode_returns),
            "evaluation/episode_length_std": pstdev(self._episode_lengths),
            "evaluation/episodes": float(self.completed_episodes),
            "evaluation/reports": float(iteration),
            "evaluation/elapsed_time": elapsed_time,
        }
        metrics.update({f"Episode/{name}": fmean(values) for name, values in self._episode_metrics.items() if values})
        for name, value in self._direct_metrics.items():
            metrics[f"{name}/frame"] = value
            metrics[f"{name}/iter"] = value
            metrics[f"{name}/time"] = value
        return metrics

    def summary(self, iteration: int, elapsed_time: float) -> dict[str, float]:
        """Return the final evaluation report after every environment reaches its target."""
        if not self.is_complete:
            raise RuntimeError("Cannot summarize evaluation before every environment completes its episodes.")
        return self.report(iteration, elapsed_time)

    def _record_metrics(self, infos: dict[str, Any]) -> None:
        """Retain metrics using the same episode/direct split as ``IsaacAlgoObserver``."""
        episode_info = infos.get("episode")
        if isinstance(episode_info, dict):
            for name, value in episode_info.items():
                scalar = _as_scalar(value)
                if scalar is not None:
                    self._episode_metrics[name].append(scalar)

        self._direct_metrics = {}
        for name, value in infos.items():
            if name in {"episode", "time_outs", "final_obs"}:
                continue
            scalar = _as_scalar(value)
            if scalar is not None:
                self._direct_metrics[name] = scalar


def _as_scalar(value: Any) -> float | None:
    """Convert a scalar-like metric value to a Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _reset_rnn_states(agent: Any, dones: torch.Tensor) -> None:
    """Clear recurrent player state for environments that just completed an episode."""
    if getattr(agent, "is_rnn", False) and getattr(agent, "states", None) is not None:
        for state in agent.states:
            state[:, dones, :] = 0.0


def _rollout(
    env: Any,
    agent: Any,
    num_episodes: int,
    on_completed_episodes: Callable[[EpisodeAccumulator], None] | None = None,
) -> EpisodeAccumulator:
    """Run an inference-only rollout until each environment completes its target episodes.

    Args:
        env: RL-Games-compatible vectorized environment.
        agent: Restored RL-Games player.
        num_episodes: Completed trajectories to retain per environment.
        on_completed_episodes: Optional callback invoked after every retained completion batch.
    """
    accumulator = EpisodeAccumulator(env.unwrapped.num_envs, num_episodes)
    observations = env.reset()
    if isinstance(observations, dict):
        observations = observations["obs"]
    _ = agent.get_batch_size(observations, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while not accumulator.is_complete:
        with torch.inference_mode():
            observations = agent.obs_to_torch(observations)
            actions = agent.get_action(observations, is_deterministic=agent.is_deterministic)
            observations, rewards, dones, infos = env.step(actions)
            _reset_rnn_states(agent, dones)
            if accumulator.update(rewards, dones, infos) and on_completed_episodes is not None:
                on_completed_episodes(accumulator)

    return accumulator


def _load_player(runner_cls: type, agent_cfg: dict[str, Any], checkpoint: str, num_actors: int) -> Any:
    """Create and restore an RL-Games player without invoking a training algorithm."""
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = num_actors

    runner = runner_cls()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(checkpoint)
    player.reset()
    return player


def _start_wandb_run(
    args_cli: argparse.Namespace,
    log_dir: str,
    env_cfg: Any,
    agent_cfg: dict[str, Any],
    checkpoint: str,
) -> Any | None:
    """Create the independent WandB evaluation run when tracking is enabled."""
    if not args_cli.track:
        return None
    if args_cli.wandb_entity is None:
        raise ValueError("Weights and Biases entity must be specified for tracking.")

    import wandb

    config_name = agent_cfg["params"]["config"]["name"]
    run = wandb.init(
        project=config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name,
        entity=args_cli.wandb_entity,
        name=os.path.basename(log_dir) if args_cli.wandb_name is None else args_cli.wandb_name,
        # Evaluation sends each report through run.log() so W&B history is available immediately.
        # Keep the local TensorBoard event file, but do not upload it a second time through the watcher.
        sync_tensorboard=False,
        monitor_gym=True,
        save_code=True,
        dir=log_dir,
    )
    if not run.resumed:
        run.config.update(
            {
                "env_cfg": env_cfg.to_dict(),
                "agent_cfg": agent_cfg,
                "evaluation": {
                    "checkpoint": checkpoint,
                    "num_episodes_per_env": args_cli.num_episodes,
                    "num_envs": env_cfg.scene.num_envs,
                    "seed": env_cfg.seed,
                },
            }
        )
    return run


def _tensorboard_step(metric_name: str, frames: int, iteration: int, elapsed_time: float) -> int:
    """Select the TensorBoard x-axis used by the matching RL-Games training metric."""
    if metric_name.startswith("Episode/") or metric_name.endswith("/iter") or metric_name.startswith("evaluation/"):
        return iteration
    if metric_name.endswith("/time"):
        return int(elapsed_time)
    return frames


def _write_report(
    writer: Any,
    wandb_run: Any | None,
    metrics: dict[str, float],
    frames: int,
    iteration: int,
    elapsed_time: float,
) -> None:
    """Write one evaluation report to local TensorBoard and optional W&B history."""
    for metric_name, value in metrics.items():
        writer.add_scalar(metric_name, value, _tensorboard_step(metric_name, frames, iteration, elapsed_time))
    writer.add_scalar("global_step", frames, iteration)
    writer.flush()

    if wandb_run is not None:
        wandb_run.log({**metrics, "global_step": frames}, step=frames)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse inference-only RL-Games evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an RL-Games checkpoint without training.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
    parser.add_argument(
        "--video_length",
        type=_positive_int,
        default=None,
        help="Length of each recorded video clip in env steps. Overrides the value in VideoRecorderCfg.",
    )
    parser.add_argument(
        "--video_interval",
        type=_positive_int,
        default=None,
        help="Interval between video clips in env steps. Overrides the value in VideoRecorderCfg.",
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=_positive_int, default=None, help="Number of environments to simulate.")
    parser.add_argument(
        "--num_episodes", type=_positive_int, required=True, help="Completed episodes to evaluate per env."
    )
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint path to load read-only for evaluation."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--train_env_cfg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the training environment configuration as-is (default: true).",
    )
    parser.add_argument("--wandb-project-name", type=str, default=None, help="Weights and Biases project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights and Biases entity (team) name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Weights and Biases evaluation run name.")
    parser.add_argument(
        "--track",
        type=lambda value: bool(strtobool(value)),
        default=False,
        nargs="?",
        const=True,
        help="Track this independent evaluation run with Weights and Biases.",
    )
    add_launcher_args(parser)
    add_frontend_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv, agent_library="rl_games")
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def run(argv: list[str]) -> None:
    """Evaluate an RL-Games checkpoint without creating or updating a policy optimizer."""
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    pre_launch_video_config(env_cfg, args_cli=args_cli)

    with startup_screen(args_cli, num_stages=3) as screen:
        show_run_summary(screen, args_cli, env_cfg, library="rl_games", action="evaluate")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            apply_env_overrides(args_cli, env_cfg)
            if args_cli.seed == -1:
                args_cli.seed = random.randint(0, 10000)
            agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
            env_cfg.seed = agent_cfg["params"]["seed"]

            checkpoint = retrieve_file_path(args_cli.checkpoint)
            config_name = agent_cfg["params"]["config"]["name"]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.abspath(os.path.join("logs", "rl_games", config_name, "evaluations", timestamp))
            write_run_manifest(
                log_dir,
                library="rl_games",
                task=args_cli.task,
                metadata={"agent": args_cli.agent, "checkpoint": checkpoint, "run_type": "evaluation"},
            )
            dump_train_configs(log_dir, env_cfg, agent_cfg)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="evaluation")

            rl_device = agent_cfg["params"]["config"]["device"]
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            obs_groups = agent_cfg["params"]["env"].get("obs_groups")
            concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
            )

            screen.stage("Loading policy")
            if args_cli.deterministic:
                # Configure after Runner construction, matching the training and play entrypoints.
                configure_seed(env_cfg.seed, torch_deterministic=True)
            player = _load_player(Runner, agent_cfg, checkpoint, env.unwrapped.num_envs)

            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)
            wandb_run = None
            try:
                wandb_run = _start_wandb_run(args_cli, log_dir, env_cfg, agent_cfg, checkpoint)
                screen.close()
                rollout_start_time = time.perf_counter()
                report_iteration = 0

                def write_completed_episode_report(accumulator: EpisodeAccumulator) -> None:
                    nonlocal report_iteration
                    report_iteration += 1
                    elapsed_time = time.perf_counter() - rollout_start_time
                    metrics = accumulator.report(report_iteration, elapsed_time)
                    _write_report(
                        writer,
                        wandb_run,
                        metrics,
                        accumulator.frames,
                        report_iteration,
                        elapsed_time,
                    )
                    print(
                        "[INFO] Evaluation progress: "
                        f"{accumulator.completed_episodes} completed episodes, "
                        f"{accumulator.frames} environment transitions."
                    )

                accumulator = _rollout(
                    env,
                    player,
                    args_cli.num_episodes,
                    on_completed_episodes=write_completed_episode_report,
                )
                elapsed_time = time.perf_counter() - rollout_start_time
                summary = accumulator.summary(report_iteration, elapsed_time)
                if wandb_run is not None:
                    wandb_run.summary.update({**summary, "global_step": accumulator.frames})
                print("[INFO] Evaluation summary:")
                for metric_name, value in summary.items():
                    print(f"  {metric_name}: {value}")
            finally:
                writer.close()
                if wandb_run is not None:
                    wandb_run.finish()
                env.close()
