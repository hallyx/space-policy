"""Train-only labels and head for the B3 action-conditioned base auxiliary.

This module is intentionally narrower than ``reaction_model.py``:

* it has no recurrent/casual rollout API and no R0 checkpoint;
* the learned head receives only an emitted candidate action micro-chunk;
* canonical base diagnostics are used only to derive a training target; and
* target pose, end-effector pose, current base state, and future state are
  never policy-model inputs.

The output target follows the same documented geometry convention as the
Reaction path so B3/B4 comparisons cannot accidentally differ only by frame:
``T_base(t)_from_base(t+K)`` as an SE(3) tangent and world twist at ``t+K``
expressed in that future base frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .geometry import (
    quaternion_geodesic_distance_xyzw,
    quaternion_inverse_xyzw,
    quaternion_rotate_vector_xyzw,
    relative_pose_body_xyzw,
    se3_exp_xyzw,
    se3_log_xyzw,
)


BASE_AUXILIARY_LABEL_SCHEMA_VERSION = "body_delta_se3_future_body_twist_v1"
BASE_AUXILIARY_DIAGNOSTIC_KEYS = (
    "diagnostics.base_pose",
    "diagnostics.base_velocity",
)
BASE_AUXILIARY_POSE_WIDTH = 7
BASE_AUXILIARY_TWIST_WIDTH = 6
BASE_AUXILIARY_PREDICTION_WIDTH = 12


def _as_finite_tensor(value: Any, *, name: str, shape: tuple[int, ...], reference: Tensor | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    if reference is not None:
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    if tuple(tensor.shape) != shape or not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} must be finite with shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _as_padding_mask(
    value: Any | None,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if value is None:
        return torch.zeros(batch_size, 2, dtype=torch.bool, device=device)
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if tensor.dtype != torch.bool or tensor.shape != (batch_size, 2):
        raise ValueError(f"{name} must be bool with shape ({batch_size}, 2), got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class BaseAuxiliaryLabels:
    """B3 labels derived from diagnostic windows, never an online observation."""

    target_body_delta_se3: Tensor
    target_future_twist_body: Tensor
    valid: Tensor


@dataclass(frozen=True)
class BaseAuxiliaryPrediction:
    """One non-recurrent base prediction from a candidate action micro-chunk."""

    body_delta_se3: Tensor
    future_twist_body: Tensor


@dataclass(frozen=True)
class BaseAuxiliaryLoss:
    """Named B3 loss terms, with a zero-safe mask for episode boundaries."""

    total: Tensor
    translation: Tensor
    rotation: Tensor
    twist: Tensor
    valid_samples: Tensor


def derive_base_auxiliary_labels(
    base_pose_window: Any,
    base_velocity_window: Any,
    *,
    base_pose_is_pad: Any | None = None,
    base_velocity_is_pad: Any | None = None,
    reference: Tensor | None = None,
) -> BaseAuxiliaryLabels:
    """Derive a K-step label from exactly ``[t, t+K]`` diagnostics.

    The caller supplies a temporal window rather than a future state object.
    It is rejected unless both entries have the exact expected shape, making it
    impossible to accidentally pass a longer action horizon and choose an
    unreviewed future offset.  Padding marks labels invalid instead of treating
    LeRobot's edge-clamped values as physical future observations.
    """

    pose_tensor = base_pose_window if isinstance(base_pose_window, Tensor) else torch.as_tensor(base_pose_window)
    if pose_tensor.ndim != 3:
        raise ValueError(
            "diagnostics.base_pose must be a [B, 2, 7] train-only label window, "
            f"got {tuple(pose_tensor.shape)}"
        )
    batch_size = pose_tensor.shape[0]
    pose = _as_finite_tensor(
        pose_tensor,
        name="diagnostics.base_pose",
        shape=(batch_size, 2, BASE_AUXILIARY_POSE_WIDTH),
        reference=reference,
    )
    twist = _as_finite_tensor(
        base_velocity_window,
        name="diagnostics.base_velocity",
        shape=(batch_size, 2, BASE_AUXILIARY_TWIST_WIDTH),
        reference=pose,
    )
    pose_is_pad = _as_padding_mask(
        base_pose_is_pad,
        name="diagnostics.base_pose_is_pad",
        batch_size=batch_size,
        device=pose.device,
    )
    twist_is_pad = _as_padding_mask(
        base_velocity_is_pad,
        name="diagnostics.base_velocity_is_pad",
        batch_size=batch_size,
        device=pose.device,
    )
    valid = ~(pose_is_pad.any(dim=1) | twist_is_pad.any(dim=1))

    # LeRobot normally edge-clamps padded values, but do not rely on their
    # quaternion validity: an invalid label must be harmless even in a synthetic
    # or alternate loader.  Its loss is masked below, and an identity surrogate
    # avoids running geometry on an all-zero padded quaternion.
    safe_pose = pose.clone()
    if bool((~valid).any()):
        safe_pose[~valid] = 0.0
        safe_pose[~valid, :, 6] = 1.0
    body_delta_pose = relative_pose_body_xyzw(safe_pose[:, 0], safe_pose[:, 1])
    target_body_delta_se3 = se3_log_xyzw(body_delta_pose)
    base_from_world = quaternion_inverse_xyzw(safe_pose[:, 1, 3:])
    future_world_twist = twist[:, 1]
    target_future_twist_body = torch.cat(
        (
            quaternion_rotate_vector_xyzw(base_from_world, future_world_twist[:, :3]),
            quaternion_rotate_vector_xyzw(base_from_world, future_world_twist[:, 3:]),
        ),
        dim=-1,
    )
    return BaseAuxiliaryLabels(
        target_body_delta_se3=target_body_delta_se3,
        target_future_twist_body=target_future_twist_body,
        valid=valid,
    )


class ActionConditionedBaseAuxiliaryHead(nn.Module):
    """Predict a K-step base target from a generated Left15 action prefix.

    It deliberately accepts no state, RGB, target, end-effector, or diagnostic
    argument.  Visual conditioning reaches the head only through the visual
    ACT-generated candidate controls.  Its prediction is output-only: callers
    must not pass it back to an action decoder.
    """

    def __init__(self, *, action_dim: int, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        if action_dim != 15:
            raise ValueError(f"B3 base auxiliary requires canonical action15, got {action_dim}")
        if horizon <= 0 or hidden_dim <= 0:
            raise ValueError("horizon and hidden_dim must be positive")
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.network = nn.Sequential(
            nn.Linear(2 * self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, BASE_AUXILIARY_PREDICTION_WIDTH),
        )
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        # Near-zero base movement is a stable start, but nonzero action encoder
        # weights preserve a differentiable action-conditioned path immediately.
        nn.init.zeros_(final_layer.bias)
        nn.init.normal_(final_layer.weight, mean=0.0, std=1.0e-3)

    def forward(self, action_micro_chunk: Tensor | Any) -> BaseAuxiliaryPrediction:
        action = action_micro_chunk if isinstance(action_micro_chunk, Tensor) else torch.as_tensor(action_micro_chunk)
        if not action.is_floating_point():
            action = action.to(dtype=torch.get_default_dtype())
        expected_tail = (self.horizon, self.action_dim)
        if action.ndim != 3 or tuple(action.shape[1:]) != expected_tail:
            raise ValueError(
                "action_micro_chunk must have shape "
                f"[B, {self.horizon}, {self.action_dim}], got {tuple(action.shape)}"
            )
        if not bool(torch.all(torch.isfinite(action))):
            raise ValueError("action_micro_chunk must contain only finite values")
        summary = torch.cat((action.mean(dim=1), action[:, -1]), dim=-1)
        prediction = self.network(summary)
        return BaseAuxiliaryPrediction(
            body_delta_se3=prediction[:, :6],
            future_twist_body=prediction[:, 6:],
        )


def _masked_mean(per_sample: Tensor, valid: Tensor) -> Tensor:
    if per_sample.ndim != 1 or valid.shape != per_sample.shape or valid.dtype != torch.bool:
        raise ValueError("masked loss expects matching [B] values and bool validity mask")
    denominator = valid.sum()
    # A final batch may consist only of edge-clamped labels.  Returning an
    # on-device differentiable zero is safer than letting fake future labels
    # influence the head or emitting NaNs into a long training run.
    if int(denominator.item()) == 0:
        return per_sample.sum() * 0.0
    return (per_sample * valid.to(dtype=per_sample.dtype)).sum() / denominator.to(per_sample.dtype)


def base_auxiliary_loss(
    prediction: BaseAuxiliaryPrediction,
    labels: BaseAuxiliaryLabels,
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    twist_weight: float = 1.0,
) -> BaseAuxiliaryLoss:
    """Compare B3 output with diagnostics that were kept label-only."""

    if min(translation_weight, rotation_weight, twist_weight) < 0.0:
        raise ValueError("base auxiliary loss weights must be non-negative")
    predicted_delta = prediction.body_delta_se3
    predicted_twist = prediction.future_twist_body
    if (
        predicted_delta.ndim != 2
        or predicted_delta.shape[1] != 6
        or predicted_twist.shape != (predicted_delta.shape[0], 6)
    ):
        raise ValueError("base auxiliary prediction must contain [B, 6] delta and twist")
    target_delta = labels.target_body_delta_se3.to(
        device=predicted_delta.device, dtype=predicted_delta.dtype
    )
    target_twist = labels.target_future_twist_body.to(
        device=predicted_delta.device, dtype=predicted_delta.dtype
    )
    valid = labels.valid.to(device=predicted_delta.device)
    if (
        target_delta.shape != predicted_delta.shape
        or target_twist.shape != predicted_twist.shape
        or valid.shape != predicted_delta.shape[:1]
        or valid.dtype != torch.bool
    ):
        raise ValueError("base auxiliary labels must match the prediction batch and dimensions")
    predicted_pose = se3_exp_xyzw(predicted_delta)
    target_pose = se3_exp_xyzw(target_delta)
    translation = _masked_mean((predicted_pose[:, :3] - target_pose[:, :3]).square().mean(dim=-1), valid)
    rotation = _masked_mean(
        quaternion_geodesic_distance_xyzw(predicted_pose[:, 3:], target_pose[:, 3:]).square(),
        valid,
    )
    twist = _masked_mean((predicted_twist - target_twist).square().mean(dim=-1), valid)
    total = translation_weight * translation + rotation_weight * rotation + twist_weight * twist
    return BaseAuxiliaryLoss(
        total=total,
        translation=translation,
        rotation=rotation,
        twist=twist,
        valid_samples=valid.sum(),
    )


__all__ = [
    "BASE_AUXILIARY_DIAGNOSTIC_KEYS",
    "BASE_AUXILIARY_LABEL_SCHEMA_VERSION",
    "BASE_AUXILIARY_POSE_WIDTH",
    "BASE_AUXILIARY_PREDICTION_WIDTH",
    "BASE_AUXILIARY_TWIST_WIDTH",
    "ActionConditionedBaseAuxiliaryHead",
    "BaseAuxiliaryLabels",
    "BaseAuxiliaryLoss",
    "BaseAuxiliaryPrediction",
    "base_auxiliary_loss",
    "derive_base_auxiliary_labels",
]
