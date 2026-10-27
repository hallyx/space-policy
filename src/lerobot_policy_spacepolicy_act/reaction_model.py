"""Action-conditioned learned transition model for free-floating reaction rollout.

This module is intentionally independent from ACT and contains no image or task
features.  Its input state is the raw, canonical physical contract:

``base pose7 + world twist6 + measured q26 + measured dq26 + previous Left15``.

Candidate Left15 trajectories are a required separate input.  The model predicts
a body-frame SE(3) increment and state residuals; base-pose integration therefore
never adds quaternion components or Euler angles directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .geometry import (
    apply_body_delta_xyzw,
    quaternion_geodesic_distance_xyzw,
    quaternion_inverse_xyzw,
    quaternion_rotate_vector_xyzw,
    se3_exp_xyzw,
)
from .reaction_state import REACTION_MODEL_STATE_DIM


BASE_POSE_SLICE = slice(0, 7)
BASE_TWIST_SLICE = slice(7, 13)
JOINT_POSITION_SLICE = slice(13, 39)
JOINT_VELOCITY_SLICE = slice(39, 65)
PREVIOUS_ACTION_SLICE = slice(65, 80)
REACTION_STATE_DIM = REACTION_MODEL_STATE_DIM
LEFT15_ACTION_DIM = 15
LEFT_ARM_INDICES = (0, 2, 4, 6, 8, 10, 12)


def _finite_tensor(value: Tensor | Any, *, name: str, last_dim: int) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    if tensor.ndim < 2 or tensor.shape[-1] != last_dim:
        raise ValueError(f"{name} must have shape [B, ..., {last_dim}], got {tuple(tensor.shape)}")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True)
class ReactionPrediction:
    """One learned transition prediction in canonical physical coordinates."""

    next_robot_state: Tensor
    next_base_pose_world_xyzw: Tensor
    next_base_twist_world: Tensor
    body_delta_se3: Tensor
    uncertainty_log_variance: Tensor | None = None


@dataclass(frozen=True)
class ReactionLoss:
    """Named, differentiable loss terms for independent reaction training."""

    total: Tensor
    position: Tensor
    rotation: Tensor
    twist: Tensor
    joint: Tensor


@dataclass(frozen=True)
class ReactionBaseLoss:
    """Leak-free v1 supervision terms available from ReactionTransition labels."""

    total: Tensor
    translation: Tensor
    rotation: Tensor
    twist: Tensor


class ReactionStateNormalizer(nn.Module):
    """A checkpointed affine normalizer for raw reaction-model inputs.

    ACT's policy processor normalizes its own observation/action tensors.  This
    module owns the raw physical reaction-state normalization so the separately
    trained transition model has the same behavior online and offline.  All
    buffers default to identity; ``set_statistics`` is called by the standalone
    reaction-training entrypoint before a checkpoint is sealed.
    """

    def __init__(self, state_dim: int = REACTION_STATE_DIM, action_dim: int = LEFT15_ACTION_DIM) -> None:
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    @torch.no_grad()
    def set_statistics(
        self,
        *,
        state_mean: Tensor | Any,
        state_std: Tensor | Any,
        action_mean: Tensor | Any,
        action_std: Tensor | Any,
    ) -> None:
        values = {
            "state_mean": torch.as_tensor(state_mean, dtype=self.state_mean.dtype, device=self.state_mean.device),
            "state_std": torch.as_tensor(state_std, dtype=self.state_std.dtype, device=self.state_std.device),
            "action_mean": torch.as_tensor(action_mean, dtype=self.action_mean.dtype, device=self.action_mean.device),
            "action_std": torch.as_tensor(action_std, dtype=self.action_std.dtype, device=self.action_std.device),
        }
        expected = {
            "state_mean": self.state_mean.shape,
            "state_std": self.state_std.shape,
            "action_mean": self.action_mean.shape,
            "action_std": self.action_std.shape,
        }
        for name, value in values.items():
            if value.shape != expected[name] or not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{name} must be finite with shape {tuple(expected[name])}")
        if bool(torch.any(values["state_std"] <= 0.0)) or bool(torch.any(values["action_std"] <= 0.0)):
            raise ValueError("reaction normalizer standard deviations must be positive")
        self.state_mean.copy_(values["state_mean"])
        self.state_std.copy_(values["state_std"])
        self.action_mean.copy_(values["action_mean"])
        self.action_std.copy_(values["action_std"])

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std.clamp_min(1.0e-6)

    def normalize_action(self, action: Tensor) -> Tensor:
        return (action - self.action_mean) / self.action_std.clamp_min(1.0e-6)


class LearnedReactionModel(nn.Module):
    """Learned action-conditioned state transition with a causal rollout API.

    It deliberately receives neither images nor target/task pose.  ``joint_trajectory``
    is encoded from all of its micro-step commands, then combined with the current
    physical state.  This makes it impossible for a caller to accidentally use an
    observation-only base-error predictor in place of a reaction model.
    """

    def __init__(
        self,
        *,
        state_dim: int = REACTION_STATE_DIM,
        action_dim: int = LEFT15_ACTION_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if state_dim != REACTION_STATE_DIM:
            raise ValueError(
                f"phase-one reaction state_dim must be {REACTION_STATE_DIM}, got {state_dim}"
            )
        if action_dim != LEFT15_ACTION_DIM:
            raise ValueError(
                f"phase-one reaction action_dim must be {LEFT15_ACTION_DIM}, got {action_dim}"
            )
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_uncertainty = bool(predict_uncertainty)
        self.normalizer = ReactionStateNormalizer(state_dim=state_dim, action_dim=action_dim)

        self.action_token_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        input_dim = state_dim + hidden_dim * 2
        for layer_index in range(num_layers):
            layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.transition_trunk = nn.Sequential(*layers)
        # [body SE3(6), world twist residual(6), q residual(26), dq residual(26)]
        self.transition_head = nn.Linear(hidden_dim, 64)
        self.uncertainty_head = nn.Linear(hidden_dim, 12) if self.predict_uncertainty else None

        # A nonzero, action-dependent initialization makes action conditioning
        # observable before fitting, while retaining a near-identity transition.
        nn.init.zeros_(self.transition_head.bias)
        nn.init.normal_(self.transition_head.weight, mean=0.0, std=1.0e-3)

    def _encode_trajectory(self, trajectory: Tensor) -> Tensor:
        encoded_tokens = self.action_token_encoder(self.normalizer.normalize_action(trajectory))
        return torch.cat((encoded_tokens.mean(dim=1), encoded_tokens[:, -1]), dim=-1)

    def forward(
        self,
        robot_state: Tensor | Any,
        joint_trajectory: Tensor | Any,
        dynamics_context: Tensor | None = None,
    ) -> ReactionPrediction:
        """Predict one transition from a raw state and a candidate Left15 path.

        ``dynamics_context`` is reserved for future payload/inertia context.  It
        must currently be ``None`` so a phase-one checkpoint cannot silently use
        an unknown privileged feature.
        """

        if dynamics_context is not None:
            raise NotImplementedError("phase-one learned_transition does not consume dynamics_context")
        state = _finite_tensor(robot_state, name="robot_state", last_dim=self.state_dim)
        trajectory = _finite_tensor(joint_trajectory, name="joint_trajectory", last_dim=self.action_dim)
        if trajectory.ndim != 3:
            raise ValueError(
                "joint_trajectory must have shape [B, micro_horizon, action15], "
                f"got {tuple(trajectory.shape)}"
            )
        if state.shape[0] != trajectory.shape[0]:
            raise ValueError("robot_state and joint_trajectory batch dimensions must match")
        if trajectory.device != state.device or trajectory.dtype != state.dtype:
            trajectory = trajectory.to(device=state.device, dtype=state.dtype)

        latent = self.transition_trunk(
            torch.cat((self.normalizer.normalize_state(state), self._encode_trajectory(trajectory)), dim=-1)
        )
        residuals = self.transition_head(latent)
        body_delta_se3 = residuals[:, :6]
        twist_residual = residuals[:, 6:12]
        joint_position_residual = residuals[:, 12:38]
        joint_velocity_residual = residuals[:, 38:64]

        current_pose = state[:, BASE_POSE_SLICE]
        next_pose = apply_body_delta_xyzw(current_pose, se3_exp_xyzw(body_delta_se3))
        next_twist = state[:, BASE_TWIST_SLICE] + twist_residual
        next_joint_position = state[:, JOINT_POSITION_SLICE] + joint_position_residual
        next_joint_velocity = state[:, JOINT_VELOCITY_SLICE] + joint_velocity_residual

        # The directly actuated arm has a candidate-command baseline.  The learned
        # residual remains free to model tracking lag and coupling, while the model
        # cannot become action-independent by construction.
        final_action = trajectory[:, -1]
        next_joint_position = next_joint_position.clone()
        next_joint_velocity = next_joint_velocity.clone()
        arm_index = torch.tensor(LEFT_ARM_INDICES, device=state.device)
        next_joint_position[:, arm_index] = final_action[:, :7] + joint_position_residual[:, arm_index]
        next_joint_velocity[:, arm_index] = final_action[:, 7:14] + joint_velocity_residual[:, arm_index]
        next_state = torch.cat(
            (
                next_pose,
                next_twist,
                next_joint_position,
                next_joint_velocity,
                final_action,
            ),
            dim=-1,
        )
        uncertainty = self.uncertainty_head(latent) if self.uncertainty_head is not None else None
        return ReactionPrediction(
            next_robot_state=next_state,
            next_base_pose_world_xyzw=next_pose,
            next_base_twist_world=next_twist,
            body_delta_se3=body_delta_se3,
            uncertainty_log_variance=uncertainty,
        )


def reaction_transition_loss(
    prediction: ReactionPrediction,
    target_robot_state: Tensor | Any,
    *,
    pose_weight: float = 1.0,
    twist_weight: float = 1.0,
    joint_weight: float = 1.0,
) -> ReactionLoss:
    """Compute one-step physical-state supervision without future-input leakage."""

    if min(pose_weight, twist_weight, joint_weight) < 0.0:
        raise ValueError("reaction loss weights must be non-negative")
    target = _finite_tensor(target_robot_state, name="target_robot_state", last_dim=REACTION_STATE_DIM)
    if target.shape != prediction.next_robot_state.shape:
        raise ValueError(
            "target_robot_state must match prediction shape: "
            f"{tuple(target.shape)} vs {tuple(prediction.next_robot_state.shape)}"
        )
    position = torch.mean((prediction.next_base_pose_world_xyzw[:, :3] - target[:, :3]).square())
    rotation = torch.mean(
        quaternion_geodesic_distance_xyzw(
            prediction.next_base_pose_world_xyzw[:, 3:], target[:, 3:7]
        ).square()
    )
    twist = torch.mean((prediction.next_base_twist_world - target[:, BASE_TWIST_SLICE]).square())
    joint = torch.mean(
        (prediction.next_robot_state[:, JOINT_POSITION_SLICE] - target[:, JOINT_POSITION_SLICE]).square()
    ) + torch.mean(
        (prediction.next_robot_state[:, JOINT_VELOCITY_SLICE] - target[:, JOINT_VELOCITY_SLICE]).square()
    )
    total = pose_weight * (position + rotation) + twist_weight * twist + joint_weight * joint
    return ReactionLoss(total=total, position=position, rotation=rotation, twist=twist, joint=joint)


def reaction_base_transition_loss(
    prediction: ReactionPrediction,
    *,
    target_delta_pose_body_xyzw: Tensor | Any,
    target_future_twist_body: Tensor | Any,
    pose_weight: float = 1.0,
    twist_weight: float = 1.0,
) -> ReactionBaseLoss:
    """Train v1 dynamics from its non-privileged base-transition labels.

    ``ReactionTransition.training_target`` intentionally exposes only the body
    delta pose and future base twist.  This loss consumes exactly those labels;
    it does not reconstruct a target from future task/EE diagnostics or feed any
    future state into the model input.
    """

    if pose_weight < 0.0 or twist_weight < 0.0:
        raise ValueError("pose_weight and twist_weight must be non-negative")
    target_pose = _finite_tensor(
        target_delta_pose_body_xyzw,
        name="target_delta_pose_body_xyzw",
        last_dim=7,
    ).to(device=prediction.body_delta_se3.device, dtype=prediction.body_delta_se3.dtype)
    target_twist_body = _finite_tensor(
        target_future_twist_body,
        name="target_future_twist_body",
        last_dim=6,
    ).to(device=prediction.body_delta_se3.device, dtype=prediction.body_delta_se3.dtype)
    if target_pose.shape[0] != prediction.body_delta_se3.shape[0] or target_twist_body.shape[0] != prediction.body_delta_se3.shape[0]:
        raise ValueError("reaction target batch dimensions must match prediction")
    predicted_delta_pose = se3_exp_xyzw(prediction.body_delta_se3)
    translation = torch.mean((predicted_delta_pose[:, :3] - target_pose[:, :3]).square())
    rotation = torch.mean(
        quaternion_geodesic_distance_xyzw(predicted_delta_pose[:, 3:], target_pose[:, 3:]).square()
    )
    world_twist = prediction.next_base_twist_world
    base_from_world = quaternion_inverse_xyzw(prediction.next_base_pose_world_xyzw[:, 3:])
    predicted_twist_body = torch.cat(
        (
            quaternion_rotate_vector_xyzw(base_from_world, world_twist[:, :3]),
            quaternion_rotate_vector_xyzw(base_from_world, world_twist[:, 3:]),
        ),
        dim=-1,
    )
    twist = torch.mean((predicted_twist_body - target_twist_body).square())
    total = pose_weight * (translation + rotation) + twist_weight * twist
    return ReactionBaseLoss(total=total, translation=translation, rotation=rotation, twist=twist)


def rollout_reaction_model(
    model: LearnedReactionModel,
    initial_robot_state: Tensor,
    joint_trajectories: Tensor,
) -> tuple[Tensor, tuple[ReactionPrediction, ...]]:
    """Causally roll out ``[B, N, H, 15]`` candidate micro-chunks.

    No future ground-truth state is read after ``initial_robot_state``.  The
    returned state sequence has shape ``[B, N + 1, 80]`` and remains fully
    differentiable through both the transition model and SE(3) pose updates.
    """

    state = _finite_tensor(initial_robot_state, name="initial_robot_state", last_dim=REACTION_STATE_DIM)
    trajectories = _finite_tensor(joint_trajectories, name="joint_trajectories", last_dim=LEFT15_ACTION_DIM)
    if trajectories.ndim != 4:
        raise ValueError("joint_trajectories must have shape [B, N, H, 15]")
    if state.shape[0] != trajectories.shape[0]:
        raise ValueError("initial_robot_state and joint_trajectories batch dimensions must match")
    predictions: list[ReactionPrediction] = []
    states = [state]
    current = state
    for index in range(trajectories.shape[1]):
        prediction = model(current, trajectories[:, index])
        predictions.append(prediction)
        current = prediction.next_robot_state
        states.append(current)
    return torch.stack(states, dim=1), tuple(predictions)


__all__ = [
    "BASE_POSE_SLICE",
    "BASE_TWIST_SLICE",
    "JOINT_POSITION_SLICE",
    "JOINT_VELOCITY_SLICE",
    "PREVIOUS_ACTION_SLICE",
    "REACTION_STATE_DIM",
    "LEFT15_ACTION_DIM",
    "LEFT_ARM_INDICES",
    "ReactionLoss",
    "ReactionBaseLoss",
    "ReactionPrediction",
    "ReactionStateNormalizer",
    "LearnedReactionModel",
    "reaction_transition_loss",
    "reaction_base_transition_loss",
    "rollout_reaction_model",
]
