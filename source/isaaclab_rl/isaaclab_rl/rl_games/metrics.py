# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Completed-episode metrics shared by RL-Games training and evaluation."""

import subprocess
from pathlib import Path
from typing import Any

import torch


class _ZAxisAlignMetrics:
    """Average the independent alignment channel by completed episode count."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, payload: dict[str, Any] | None, retained_dones: torch.Tensor | None = None) -> None:
        """Include per-episode values, optionally filtering retained environment IDs."""
        if payload is None:
            return
        env_ids = payload["env_ids"].detach().long()
        keep = None
        if retained_dones is not None:
            keep = retained_dones[env_ids.to(retained_dones.device)]
        for name, values in payload["metrics"].items():
            values = values.detach().flatten()
            if values.numel() != env_ids.numel():
                raise ValueError("Z-axis metrics must contain one value per completed environment ID.")
            if keep is not None:
                values = values[keep.to(values.device)]
            if values.numel() > 0:
                self._sums[name] = self._sums.get(name, 0.0) + values.double().sum().item()
                self._counts[name] = self._counts.get(name, 0) + values.numel()

    def report(self) -> dict[str, float]:
        """Return the means with exactly one namespace prefix."""
        return {f"ZAxisAlign/{name}": total / self._counts[name] for name, total in self._sums.items()}

    def clear(self) -> None:
        """Begin a new reporting window."""
        self._sums.clear()
        self._counts.clear()


def _rl_games_run_metadata(env: Any, checkpoint: str | None) -> dict[str, str]:
    """Capture source provenance and the independent RNG seed without changing Git state."""
    metadata = {"checkpoint": checkpoint or ""}
    root = Path(__file__).resolve().parents[4]
    for key, command in (
        ("code_revision", ["git", "rev-parse", "HEAD"]),
        ("code_dirty", ["git", "status", "--porcelain"]),
    ):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True, timeout=10)
            metadata[key] = str(bool(result.stdout.strip())) if key == "code_dirty" else result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            metadata[key] = "unavailable"
    seed = getattr(env.unwrapped, "z_axis_align_seed", None)
    if seed is not None:
        metadata["z_axis_align_seed"] = str(seed)
    return metadata
