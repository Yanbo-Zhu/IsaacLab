# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RL-Games observer with a separate completed-episode alignment channel."""

from typing import Any

from rl_games.common.algo_observer import IsaacAlgoObserver

from .metrics import _ZAxisAlignMetrics


class RlGamesAlgoObserver(IsaacAlgoObserver):
    """Preserve Isaac metrics and report alignment metrics under ``ZAxisAlign/``.

    The ``z_axis_align`` info channel contains completed ``env_ids`` and a
    ``metrics`` dictionary of corresponding per-episode tensors. Metrics are
    averaged over episodes completed since the previous report, using epoch
    as the TensorBoard step. Existing episode and direct metrics retain their
    original names and reporting behavior.
    """

    def after_init(self, algo: Any) -> None:
        """Initialize the original observer and the alignment report window."""
        super().after_init(algo)
        self._z_axis_align_metrics = _ZAxisAlignMetrics()

    def process_infos(self, infos: dict[str, Any], done_indices: Any) -> None:
        """Collect original metrics and independent completed alignment episodes."""
        super().process_infos(infos, done_indices)
        self._z_axis_align_metrics.update(infos.get("z_axis_align"))

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        """Write the alignment report at the training epoch and flush the writer."""
        super().after_print_stats(frame, epoch_num, total_time)
        for name, value in self._z_axis_align_metrics.report().items():
            self.writer.add_scalar(name, value, epoch_num)
        self._z_axis_align_metrics.clear()
        self.writer.flush()

    def close(self) -> None:
        """Drain the asynchronous writer before the simulation application shuts down."""
        writer = getattr(self, "writer", None)
        if writer is not None:
            # TensorBoardX flush() does not wait for its event queue; close() does.
            writer.close()
