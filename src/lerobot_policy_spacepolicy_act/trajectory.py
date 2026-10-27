"""Batch-differentiable cubic Hermite trajectories for the Left15 arm7 action.

The public functions operate only on the seven actuated left-arm joints.  The
gripper is intentionally excluded: it retains its separately defined position
control semantics.  Positions are radians, endpoint velocities are radians per
second, and durations/sample times are seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ARM7_DIM = 7


@dataclass(frozen=True)
class Arm7CubicHermiteTrajectory:
    """A sampled arm7 trajectory with positions and time derivatives."""

    positions: torch.Tensor
    velocities: torch.Tensor
    sample_times: torch.Tensor
    duration: torch.Tensor


def _floating_tensor(value: Any, *, name: str, reference: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif reference is None:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
    else:
        result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if not result.is_floating_point():
        result = result.to(dtype=torch.get_default_dtype())
    if reference is not None and (result.device != reference.device or result.dtype != reference.dtype):
        result = result.to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _arm7_tensor(value: Any, *, name: str, reference: torch.Tensor | None = None) -> torch.Tensor:
    result = _floating_tensor(value, name=name, reference=reference)
    if result.ndim < 1 or result.shape[-1] != ARM7_DIM:
        raise ValueError(f"{name} must have shape (..., {ARM7_DIM}), got {tuple(result.shape)}")
    return result


def _broadcast_endpoints(
    start_position: Any,
    start_velocity: Any,
    end_position: Any,
    end_velocity: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start = _arm7_tensor(start_position, name="start_position")
    start_velocity_tensor = _arm7_tensor(
        start_velocity, name="start_velocity", reference=start
    )
    end = _arm7_tensor(end_position, name="end_position", reference=start)
    end_velocity_tensor = _arm7_tensor(end_velocity, name="end_velocity", reference=start)
    shape = torch.broadcast_shapes(
        start.shape[:-1],
        start_velocity_tensor.shape[:-1],
        end.shape[:-1],
        end_velocity_tensor.shape[:-1],
    )
    return (
        start.expand((*shape, ARM7_DIM)),
        start_velocity_tensor.expand((*shape, ARM7_DIM)),
        end.expand((*shape, ARM7_DIM)),
        end_velocity_tensor.expand((*shape, ARM7_DIM)),
    )


def _duration_tensor(duration: Any, *, batch_shape: torch.Size, reference: torch.Tensor) -> torch.Tensor:
    duration_tensor = _floating_tensor(duration, name="duration", reference=reference)
    try:
        duration_tensor = torch.broadcast_to(duration_tensor, batch_shape)
    except RuntimeError as error:
        raise ValueError(
            f"duration with shape {tuple(duration_tensor.shape)} cannot broadcast to batch shape {tuple(batch_shape)}"
        ) from error
    if not bool(torch.all(duration_tensor > 0.0)):
        raise ValueError("duration must be strictly positive")
    return duration_tensor


def _basis_and_derivative(normalized_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t2 = normalized_time.square()
    t3 = normalized_time * t2
    return (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + normalized_time,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
        6.0 * t2 - 6.0 * normalized_time,
        3.0 * t2 - 4.0 * normalized_time + 1.0,
        -6.0 * t2 + 6.0 * normalized_time,
        3.0 * t2 - 2.0 * normalized_time,
    )


def _evaluate_normalized(
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    end_position: torch.Tensor,
    end_velocity: torch.Tensor,
    normalized_time: torch.Tensor,
    duration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h00, h10, h01, h11, dh00, dh10, dh01, dh11 = _basis_and_derivative(normalized_time)
    duration_expanded = duration.unsqueeze(-1)
    positions = (
        h00.unsqueeze(-1) * start_position
        + h10.unsqueeze(-1) * duration_expanded * start_velocity
        + h01.unsqueeze(-1) * end_position
        + h11.unsqueeze(-1) * duration_expanded * end_velocity
    )
    velocities = (
        dh00.unsqueeze(-1) * start_position
        + dh10.unsqueeze(-1) * duration_expanded * start_velocity
        + dh01.unsqueeze(-1) * end_position
        + dh11.unsqueeze(-1) * duration_expanded * end_velocity
    ) / duration_expanded
    return positions, velocities


def evaluate_arm7_cubic_hermite(
    start_position: Any,
    start_velocity: Any,
    end_position: Any,
    end_velocity: Any,
    sample_times: Any,
    *,
    duration: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a cubic Hermite arm7 segment at one or more physical times.

    Endpoint tensors have shape ``(..., 7)`` and broadcast over their leading
    batch dimensions.  ``sample_times`` is either a scalar, yielding
    ``(..., 7)``, or a shared ``[T]`` vector, yielding ``(..., T, 7)``.
    Every requested time must be in the closed interval ``[0, duration]``.
    """

    start, start_velocity_tensor, end, end_velocity_tensor = _broadcast_endpoints(
        start_position, start_velocity, end_position, end_velocity
    )
    batch_shape = start.shape[:-1]
    duration_tensor = _duration_tensor(duration, batch_shape=batch_shape, reference=start)
    times = _floating_tensor(sample_times, name="sample_times", reference=start)
    if times.ndim > 1:
        raise ValueError("sample_times must be a scalar or a one-dimensional tensor")
    if bool(torch.any(times < 0.0)):
        raise ValueError("sample_times must be non-negative")

    if times.ndim == 0:
        if bool(torch.any(times > duration_tensor)):
            raise ValueError("sample_times must not exceed duration")
        normalized_time = times / duration_tensor
        return _evaluate_normalized(
            start,
            start_velocity_tensor,
            end,
            end_velocity_tensor,
            normalized_time,
            duration_tensor,
        )

    time_shape = (*batch_shape, times.shape[0])
    expanded_times = times.reshape((1,) * len(batch_shape) + (times.shape[0],)).expand(time_shape)
    expanded_duration = duration_tensor.unsqueeze(-1).expand(time_shape)
    if bool(torch.any(expanded_times > expanded_duration)):
        raise ValueError("sample_times must not exceed duration")
    normalized_time = expanded_times / expanded_duration
    return _evaluate_normalized(
        start.unsqueeze(-2),
        start_velocity_tensor.unsqueeze(-2),
        end.unsqueeze(-2),
        end_velocity_tensor.unsqueeze(-2),
        normalized_time,
        expanded_duration,
    )


def sample_arm7_cubic_hermite(
    start_position: Any,
    start_velocity: Any,
    end_position: Any,
    end_velocity: Any,
    *,
    duration: Any,
    num_samples: int,
    include_start: bool = True,
) -> Arm7CubicHermiteTrajectory:
    """Uniformly sample a batch of arm7 Hermite segments.

    With ``include_start=True`` the first sample is the measured start state
    and the final sample is the endpoint.  With it disabled, samples represent
    future control targets and still include the endpoint, matching a physical
    execution queue that has already observed its start state.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    start, start_velocity_tensor, end, end_velocity_tensor = _broadcast_endpoints(
        start_position, start_velocity, end_position, end_velocity
    )
    batch_shape = start.shape[:-1]
    duration_tensor = _duration_tensor(duration, batch_shape=batch_shape, reference=start)
    if include_start:
        normalized_times = torch.linspace(
            0.0, 1.0, num_samples, dtype=start.dtype, device=start.device
        )
    else:
        normalized_times = torch.arange(
            1, num_samples + 1, dtype=start.dtype, device=start.device
        ) / float(num_samples)
    normalized_times = normalized_times.reshape((1,) * len(batch_shape) + (num_samples,)).expand(
        (*batch_shape, num_samples)
    )
    expanded_duration = duration_tensor.unsqueeze(-1).expand((*batch_shape, num_samples))
    positions, velocities = _evaluate_normalized(
        start.unsqueeze(-2),
        start_velocity_tensor.unsqueeze(-2),
        end.unsqueeze(-2),
        end_velocity_tensor.unsqueeze(-2),
        normalized_times,
        expanded_duration,
    )
    return Arm7CubicHermiteTrajectory(
        positions=positions,
        velocities=velocities,
        sample_times=normalized_times * expanded_duration,
        duration=duration_tensor,
    )


__all__ = [
    "ARM7_DIM",
    "Arm7CubicHermiteTrajectory",
    "evaluate_arm7_cubic_hermite",
    "sample_arm7_cubic_hermite",
]
