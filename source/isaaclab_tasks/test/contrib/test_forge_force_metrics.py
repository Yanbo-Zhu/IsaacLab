# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FORGE per-episode force metrics."""

import torch

from isaaclab_tasks.contrib.forge.forge_utils import _EpisodeForceStatistics


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
