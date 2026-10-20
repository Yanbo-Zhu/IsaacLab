# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeTask(FactoryTask):
    """FORGE rewards, including optional directional alignment of the assets' local z axes."""

    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]
    z_axis_align_scale: float = 0.0
    """Weight of the direction-only z-axis alignment reward."""
    z_axis_align_sigma_range: list[float] = [2.0, 10.0]
    """Uniform per-episode reward decay-angle bounds [deg]; equal bounds fix the angle.

    The sampled angle is not included in actor or critic observations. At this
    axis misalignment, the unweighted reward equals exp(-1).
    """


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2
    z_axis_align_scale: float = 0.1


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
