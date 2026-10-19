# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the dedicated evaluation launcher."""

from __future__ import annotations

import os
from pathlib import Path


def test_isaaclab_2_launcher_matches_the_primary_launcher() -> None:
    """The dedicated launcher must preserve the primary launch environment contract."""
    root = Path(__file__).resolve().parents[3]
    primary = root / "isaaclab.sh"
    evaluation = root / "isaaclab-2.sh"

    assert evaluation.read_bytes() == primary.read_bytes()
    assert os.access(evaluation, os.X_OK)
