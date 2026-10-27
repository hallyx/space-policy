"""ACT temporal-ensemble variants used by inference-only diagnostics.

LeRobot's built-in :class:`ACTTemporalEnsembler` keeps at most ``chunk_size``
overlapping predictions and, for a positive coefficient, gives the oldest
prediction the largest weight.  This module adds an explicit bounded-history
implementation so an evaluation can shorten that history without changing a
checkpoint or its model outputs.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor


class WindowedACTTemporalEnsembler:
    """Combine overlapping ACT predictions over a fixed recent history.

    At control time ``t``, a chunk predicted ``age`` steps ago contributes its
    ``age``-th action.  Candidates are ordered oldest to newest and weighted as
    ``exp(-coefficient * i)``.  This exactly matches LeRobot's weighting order;
    a positive coefficient favours older predictions and a negative one
    favours newer predictions.
    """

    def __init__(
        self,
        temporal_ensemble_coeff: float,
        chunk_size: int,
        max_history: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_history <= 0 or max_history > chunk_size:
            raise ValueError("max_history must be in [1, chunk_size]")
        if not torch.isfinite(torch.tensor(float(temporal_ensemble_coeff))):
            raise ValueError("temporal_ensemble_coeff must be finite")
        self.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        self.chunk_size = int(chunk_size)
        self.max_history = int(max_history)
        self.reset()

    def reset(self) -> None:
        self._chunks: deque[Tensor] = deque(maxlen=self.max_history)

    def update(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.chunk_size:
            raise ValueError(
                "actions must have shape [batch, chunk_size, action_dim], "
                f"got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("actions contain NaN/Inf")

        self._chunks.append(actions)
        history = list(self._chunks)
        count = len(history)
        # Oldest retained prediction uses the largest action offset; the newest
        # prediction always contributes action 0.
        candidates = torch.stack(
            [chunk[:, count - index - 1] for index, chunk in enumerate(history)],
            dim=1,
        )
        weights = torch.exp(
            -self.temporal_ensemble_coeff
            * torch.arange(count, dtype=actions.dtype, device=actions.device)
        )
        weights = weights / weights.sum()
        return torch.sum(candidates * weights.view(1, count, 1), dim=1)


def install_windowed_act_temporal_ensemble(
    policy: object,
    *,
    temporal_ensemble_coeff: float,
    max_history: int,
) -> WindowedACTTemporalEnsembler:
    """Replace an ACT policy's runtime ensembler without touching weights."""

    config = getattr(policy, "config", None)
    if getattr(config, "type", None) != "act":
        raise ValueError("windowed temporal ensemble is only valid for ACT")
    chunk_size = int(config.chunk_size)
    ensembler = WindowedACTTemporalEnsembler(
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        chunk_size=chunk_size,
        max_history=max_history,
    )
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
    policy.temporal_ensembler = ensembler
    policy.reset()
    return ensembler
