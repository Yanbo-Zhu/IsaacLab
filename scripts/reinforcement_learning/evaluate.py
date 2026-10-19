# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified checkpoint evaluation executable for Isaac Lab reinforcement learning workflows."""

# Warp captures ``enable_backward`` when a module is created, which happens at import
# time, so it has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_evaluate_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning evaluation library."""
    return run_evaluate_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
