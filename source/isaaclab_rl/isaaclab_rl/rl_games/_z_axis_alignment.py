# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent FORGE episode logging, without changing RL-Games optimization."""

from typing import Any

import torch

_Z_AXIS_ALIGNMENT_DESIGN = "cube_root_excess_angle_penalty_v1"


class _ZAxisAlignmentMetrics:
    """Accumulate completed-episode values with equal weight per retained episode."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def update(self, infos: dict[str, Any], retained_env_ids: torch.Tensor) -> None:
        """Retain payload rows belonging to the supplied completed environment indices."""
        payload = infos.get("z_axis_align")
        if payload is None:
            return
        env_ids = payload["env_ids"].detach().flatten()
        retained = torch.isin(env_ids, retained_env_ids.detach().flatten().to(env_ids.device))
        count = int(retained.sum().item())
        if count == 0:
            return
        values = {}
        for name, samples in payload["metrics"].items():
            if not name.startswith("ZAxisAlign/") or samples.ndim != 1 or samples.numel() != env_ids.numel():
                raise ValueError("Z-axis metrics must have one value per completed environment and a ZAxisAlign/ tag.")
            values[name] = float(samples.detach()[retained.to(samples.device)].double().sum().item())
        for name, value in values.items():
            self._sums[name] = self._sums.get(name, 0.0) + value
        self._count += count

    def report(self) -> dict[str, float]:
        """Return means over retained episodes, or no metrics before the first completion."""
        return {name: total / self._count for name, total in self._sums.items()} if self._count else {}

    def clear(self) -> None:
        """Begin a new training reporting window."""
        self._sums.clear()
        self._count = 0


class _ZAxisAlignmentObserver:
    """Implement the RL-Games observer callbacks using its existing writer only."""

    def __init__(self) -> None:
        self._metrics = _ZAxisAlignmentMetrics()
        self._algo: Any = None
        self._writer: Any = None

    def before_init(self, base_name: str, config: dict[str, Any], experiment_name: str) -> None:
        """Leave runner initialization unchanged."""

    def after_init(self, algo: Any) -> None:
        """Reuse the algorithm writer without creating another event stream."""
        self._algo = algo
        self._writer = algo.writer

    def process_infos(self, infos: dict[str, Any], done_indices: torch.Tensor) -> None:
        """Collect completed episodes without changing the supplied metadata."""
        self._metrics.update(infos, done_indices)

    def after_steps(self) -> None:
        """Leave rollout and optimization unchanged."""

    def after_clear_stats(self) -> None:
        """Keep pending completions until they have been reported."""

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        """Write the current completion window against environment transitions."""
        if self._writer is not None:
            for name, value in self._metrics.report().items():
                self._writer.add_scalar(name, value, frame)
            self._metrics.clear()

    def close(self) -> None:
        """Write pending completions and drain the shared asynchronous writer on exit."""
        if self._writer is not None:
            self.after_print_stats(self._algo.frame, 0, 0.0)
            self._writer.close()
            self._writer = None
