# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep FORGE alignment episode metrics in their requested logging namespace."""

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


class _IsaacLabAlgoObserver(IsaacAlgoObserver):
    """Preserve upstream logging while collecting sparse ZAxisAlign episode reports."""

    def after_init(self, algo) -> None:
        super().after_init(algo)
        self._alignment_infos: list[dict[str, torch.Tensor | float]] = []

    def process_infos(self, infos: dict, done_indices: torch.Tensor) -> None:
        if isinstance(infos, dict) and isinstance(infos.get("episode"), dict):
            episode = infos["episode"]
            alignment = {name: value for name, value in episode.items() if name.startswith("ZAxisAlign/")}
            if alignment:
                self._alignment_infos.append(alignment)
            # The upstream observer assumes every buffered episode dict has the same keys.
            infos = {
                **infos,
                "episode": {name: value for name, value in episode.items() if not name.startswith("ZAxisAlign/")},
            }
        super().process_infos(infos, done_indices)

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        super().after_print_stats(frame, epoch_num, total_time)
        if self._alignment_infos:
            for name in self._alignment_infos[0]:
                values = [
                    torch.as_tensor(info[name], device=self.algo.device).float() for info in self._alignment_infos
                ]
                self.writer.add_scalar(name, torch.stack(values).mean(), epoch_num)
            self._alignment_infos.clear()
            # Persist the complete report even when a short training run exits immediately.
            self.writer.flush()
