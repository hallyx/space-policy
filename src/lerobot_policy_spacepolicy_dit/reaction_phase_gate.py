"""Fail-closed pre-contact admission checks for Reaction-Aware ACT.

The control-state value is deliberately *not* a policy observation.  It is an
execution/training admission label checked at the server and processor
boundaries, then discarded before an ACT batch or Reaction-model state is
created.  V1 has no contact-phase fallback, so an absent or out-of-range value
must reject the request rather than being inferred from images, time, or robot
state.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import torch


REACTION_CONTROL_STATE_KEY = "diagnostics.control_state"
PRECONTACT_CONTROL_STATES = frozenset((13, 14, 15))
PRECONTACT_CONTROL_STATE_SEQUENCE = tuple(sorted(PRECONTACT_CONTROL_STATES))


class ReactionAwarePhaseGateError(ValueError):
    """A request/batch is outside the only validated Reaction-Aware phase."""


def normalize_precontact_control_states(
    values: Collection[int] | np.ndarray | tuple[int, ...] | list[int],
    *,
    name: str = "precontact_control_states",
) -> tuple[int, ...]:
    """Return one finite, deduplicated integer allow-list.

    This helper is intentionally strict about booleans and fractional values;
    a phase ID is a discrete state-machine label, not a model feature.
    """

    try:
        raw_values = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a non-empty collection of integer state IDs") from error
    if not raw_values:
        raise ValueError(f"{name} must not be empty")
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integer state IDs, not bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain integer state IDs, got {value!r}") from error
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must contain finite integer state IDs, got {value!r}")
        integer = int(numeric)
        if numeric != float(integer):
            raise ValueError(f"{name} must contain finite integer state IDs, got {value!r}")
        normalized.append(integer)
    return tuple(sorted(set(normalized)))


def require_precontact_control_state(
    value: Any,
    *,
    source: str,
    allowed_control_states: Collection[int] = PRECONTACT_CONTROL_STATES,
) -> tuple[int, ...]:
    """Reject missing/malformed/non-pre-contact control-state values.

    ``value`` may be the scalar online protocol field or a batched offline
    tensor.  The return value exists solely for audit/tests; callers must not
    put it into a model input tensor.
    """

    allowed = normalize_precontact_control_states(
        allowed_control_states,
        name="allowed_control_states",
    )
    if value is None:
        raise ReactionAwarePhaseGateError(
            f"{source} is required for Reaction-Aware ACT pre-contact admission; "
            f"allowed control_state={list(allowed)}"
        )
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_complex():
            raise ReactionAwarePhaseGateError(
                f"{source} must be an integer control_state tensor, got dtype={value.dtype}"
            )
        array = value.detach().to("cpu").numpy()
    else:
        array = np.asarray(value)
    if array.size == 0 or array.ndim > 2 or (array.ndim == 2 and array.shape[-1] != 1):
        raise ReactionAwarePhaseGateError(
            f"{source} must be one scalar per sample, got shape={tuple(array.shape)}"
        )
    if array.dtype.kind == "b":
        raise ReactionAwarePhaseGateError(f"{source} must be integer control_state, not bool")
    if array.dtype.kind in {"i", "u"}:
        states = array.reshape(-1).astype(np.int64, copy=False)
    elif array.dtype.kind == "f":
        flattened = array.reshape(-1)
        if not np.all(np.isfinite(flattened)) or not np.all(flattened == np.rint(flattened)):
            raise ReactionAwarePhaseGateError(
                f"{source} must contain finite integer control_state values"
            )
        states = flattened.astype(np.int64)
    else:
        raise ReactionAwarePhaseGateError(
            f"{source} must be a numeric integer control_state, got dtype={array.dtype}"
        )
    invalid = sorted(set(int(state) for state in states if int(state) not in allowed))
    if invalid:
        raise ReactionAwarePhaseGateError(
            f"Reaction-Aware ACT has no post-contact fallback: {source} contains "
            f"control_state={invalid}, allowed={list(allowed)}"
        )
    return tuple(int(state) for state in states)


__all__ = [
    "PRECONTACT_CONTROL_STATES",
    "PRECONTACT_CONTROL_STATE_SEQUENCE",
    "REACTION_CONTROL_STATE_KEY",
    "ReactionAwarePhaseGateError",
    "normalize_precontact_control_states",
    "require_precontact_control_state",
]
