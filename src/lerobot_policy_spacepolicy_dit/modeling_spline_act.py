"""B2 Spline ACT: visual ACT proposals decoded as Hermite action15 chunks.

This module intentionally contains no reaction-state, transition-model, base
auxiliary, or future-state code.  Each invocation starts from the current
canonical state8, predicts visual ACT proposal knots, and deterministically
constructs a piecewise cubic Hermite trajectory.  The inherited ACT action
queue is sealed to one action, so every control step receives a fresh plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_spline_act import SplineACTConfig
from .trajectory import sample_arm7_cubic_hermite


_LEFT15_ACTION_DIM = 15
_ARM7_DIM = 7


def _mode_name(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


def _stats_vector(
    dataset_stats: Mapping[str, Mapping[str, Any]] | None,
    key: str,
    statistic: str,
    width: int,
    default: float,
) -> Tensor:
    if dataset_stats is None or key not in dataset_stats or statistic not in dataset_stats[key]:
        return torch.full((width,), default, dtype=torch.float32)
    value = torch.as_tensor(dataset_stats[key][statistic], dtype=torch.float32).reshape(-1)
    if value.shape != (width,) or not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"dataset_stats[{key!r}][{statistic!r}] must be finite shape ({width},)")
    return value


class SplineKnotDecoder(nn.Module):
    """Predict terminal Hermite knots from local visual ACT proposal chunks.

    The network sees only normalized ACT action proposals.  It has no raw robot
    dynamics state or privileged task field; the observed state8 is used solely
    as the physical start knot for the deterministic first segment.
    """

    def __init__(self, *, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if action_dim != _LEFT15_ACTION_DIM:
            raise ValueError(f"B2 Spline ACT expects action15, got {action_dim}")
        self.endpoint_network = nn.Sequential(
            nn.Linear(2 * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.initial_velocity_network = nn.Sequential(
            nn.Linear(2 * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, _ARM7_DIM),
        )
        for network in (self.endpoint_network, self.initial_velocity_network):
            last = network[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.bias)
            nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)

    @staticmethod
    def _summary(proposal_micro_chunks: Tensor) -> Tensor:
        if proposal_micro_chunks.ndim != 4 or proposal_micro_chunks.shape[-1] != _LEFT15_ACTION_DIM:
            raise ValueError(
                "proposal_micro_chunks must have shape [B, N, K, 15], "
                f"got {tuple(proposal_micro_chunks.shape)}"
            )
        return torch.cat(
            (proposal_micro_chunks.mean(dim=-2), proposal_micro_chunks[..., -1, :]), dim=-1
        )

    def forward(self, proposal_micro_chunks: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized terminal action knots and initial arm velocity knot."""

        summary = self._summary(proposal_micro_chunks)
        endpoint_correction = self.endpoint_network(summary)
        # Zero-initialized correction means B2 starts close to the visual ACT
        # endpoint proposal while retaining enough capacity to fit knot labels.
        endpoints = proposal_micro_chunks[..., -1, :] + 0.10 * torch.tanh(endpoint_correction)
        initial_summary = summary[:, 0]
        initial_velocity = proposal_micro_chunks[:, 0, 0, _ARM7_DIM : 2 * _ARM7_DIM]
        initial_velocity = initial_velocity + 0.10 * torch.tanh(
            self.initial_velocity_network(initial_summary)
        )
        return endpoints, initial_velocity


class SplineACTPolicy(ACTPolicy):
    """Independent B2 policy type: ACT proposal + deterministic Hermite decode."""

    config_class = SplineACTConfig
    name = "spline_act"

    def __init__(
        self,
        config: SplineACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ) -> None:
        # ACT itself does not consume stats.  B2 needs them only at its internal
        # physical-unit Hermite boundary and never exposes them as extra inputs.
        del kwargs
        super().__init__(config)
        self.config = config
        self._register_normalization_buffers(dataset_stats)
        self.knot_decoder = SplineKnotDecoder(
            action_dim=config.action_feature.shape[0],
            hidden_dim=config.knot_decoder_hidden_dim,
        )
        # Audit-only runtime traces.  They make the knot/continuity contract
        # testable without becoming model inputs or checkpoint configuration.
        self._last_spline_endpoints_raw: Tensor | None = None
        self._last_spline_start_positions_raw: Tensor | None = None
        self._last_spline_start_velocities_raw: Tensor | None = None

    def _register_normalization_buffers(
        self, dataset_stats: Mapping[str, Mapping[str, Any]] | None
    ) -> None:
        action_feature = self.config.action_feature
        state_feature = self.config.robot_state_feature
        if action_feature is None or state_feature is None:
            raise ValueError("Spline ACT requires state and action features")
        action_dim = action_feature.shape[0]
        state_dim = state_feature.shape[0]
        self.register_buffer(
            "spline_action_mean", _stats_vector(dataset_stats, ACTION, "mean", action_dim, 0.0)
        )
        self.register_buffer(
            "spline_action_std", _stats_vector(dataset_stats, ACTION, "std", action_dim, 1.0)
        )
        self.register_buffer(
            "spline_action_min", _stats_vector(dataset_stats, ACTION, "min", action_dim, -1.0)
        )
        self.register_buffer(
            "spline_action_max", _stats_vector(dataset_stats, ACTION, "max", action_dim, 1.0)
        )
        self.register_buffer(
            "spline_state_mean", _stats_vector(dataset_stats, OBS_STATE, "mean", state_dim, 0.0)
        )
        self.register_buffer(
            "spline_state_std", _stats_vector(dataset_stats, OBS_STATE, "std", state_dim, 1.0)
        )
        self.register_buffer(
            "spline_state_min", _stats_vector(dataset_stats, OBS_STATE, "min", state_dim, -1.0)
        )
        self.register_buffer(
            "spline_state_max", _stats_vector(dataset_stats, OBS_STATE, "max", state_dim, 1.0)
        )
        if bool(torch.any(self.spline_action_std <= 0.0)) or bool(
            torch.any(self.spline_state_std <= 0.0)
        ):
            raise ValueError("Spline ACT mean/std normalization requires positive std values")
        if bool(torch.any(self.spline_action_max < self.spline_action_min)) or bool(
            torch.any(self.spline_state_max < self.spline_state_min)
        ):
            raise ValueError("Spline ACT min/max normalization requires max >= min")
        self._spline_action_normalization = _mode_name(
            self.config.normalization_mapping.get("ACTION", "IDENTITY")
        )
        self._spline_state_normalization = _mode_name(
            self.config.normalization_mapping.get("STATE", "IDENTITY")
        )

    @staticmethod
    def _denormalize(
        value: Tensor,
        *,
        mode: str,
        mean: Tensor,
        std: Tensor,
        minimum: Tensor,
        maximum: Tensor,
        name: str,
    ) -> Tensor:
        if value.shape[-1] != mean.shape[0]:
            raise ValueError(
                f"{name} must end with dimension {mean.shape[0]}, got {tuple(value.shape)}"
            )
        if mode == "MEAN_STD":
            return value * std + mean
        if mode == "MIN_MAX":
            return (value + 1.0) * (maximum - minimum) / 2.0 + minimum
        if mode == "IDENTITY":
            return value
        raise ValueError(
            f"Spline ACT supports {name} normalization only in "
            f"{{MEAN_STD, MIN_MAX, IDENTITY}}, got {mode!r}"
        )

    @staticmethod
    def _normalize(
        value: Tensor,
        *,
        mode: str,
        mean: Tensor,
        std: Tensor,
        minimum: Tensor,
        maximum: Tensor,
        name: str,
    ) -> Tensor:
        if value.shape[-1] != mean.shape[0]:
            raise ValueError(
                f"{name} must end with dimension {mean.shape[0]}, got {tuple(value.shape)}"
            )
        if mode == "MEAN_STD":
            return (value - mean) / std.clamp_min(1.0e-6)
        if mode == "MIN_MAX":
            return 2.0 * (value - minimum) / (maximum - minimum).clamp_min(1.0e-6) - 1.0
        if mode == "IDENTITY":
            return value
        raise AssertionError(f"unreachable {name} normalization mode {mode!r}")

    def _denormalize_action(self, action: Tensor) -> Tensor:
        return self._denormalize(
            action,
            mode=self._spline_action_normalization,
            mean=self.spline_action_mean,
            std=self.spline_action_std,
            minimum=self.spline_action_min,
            maximum=self.spline_action_max,
            name="ACTION",
        )

    def _normalize_action(self, action: Tensor) -> Tensor:
        return self._normalize(
            action,
            mode=self._spline_action_normalization,
            mean=self.spline_action_mean,
            std=self.spline_action_std,
            minimum=self.spline_action_min,
            maximum=self.spline_action_max,
            name="ACTION",
        )

    def _denormalize_state(self, state: Tensor) -> Tensor:
        return self._denormalize(
            state,
            mode=self._spline_state_normalization,
            mean=self.spline_state_mean,
            std=self.spline_state_std,
            minimum=self.spline_state_min,
            maximum=self.spline_state_max,
            name="STATE",
        )

    def _prepare_model_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        model_batch = dict(batch)
        if self.config.image_features:
            model_batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return model_batch

    def _current_state_raw(self, batch: dict[str, Tensor], *, reference: Tensor) -> Tensor:
        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise KeyError("Spline ACT requires observation.state")
        state = state.to(device=reference.device, dtype=reference.dtype)
        if state.shape != (reference.shape[0], 8) or not bool(torch.all(torch.isfinite(state))):
            raise ValueError(
                "Spline ACT observation.state must be finite [B, 8], "
                f"got {tuple(state.shape)}"
            )
        return self._denormalize_state(state)

    def _apply_motion_limits(self, start_position: Tensor, endpoint: Tensor) -> Tensor:
        result = endpoint.clone()
        if self.config.max_joint_delta is not None:
            limit = float(self.config.max_joint_delta)
            result[:, :_ARM7_DIM] = torch.maximum(
                torch.minimum(result[:, :_ARM7_DIM], start_position + limit), start_position - limit
            )
        if self.config.max_joint_velocity is not None:
            result[:, _ARM7_DIM : 2 * _ARM7_DIM] = result[
                :, _ARM7_DIM : 2 * _ARM7_DIM
            ].clamp(min=-float(self.config.max_joint_velocity), max=float(self.config.max_joint_velocity))
        return result

    def _decode_spline_action_chunk(
        self, batch: dict[str, Tensor], proposal_normalized: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode visual ACT proposal into a continuous, physical-unit Hermite chunk.

        This is deterministic trajectory construction, not a learned state
        rollout: each later segment starts from the preceding emitted knot, and
        no hidden/predicted environment state is ever queried.
        """

        expected_shape = (self.config.chunk_size, _LEFT15_ACTION_DIM)
        if proposal_normalized.ndim != 3 or tuple(proposal_normalized.shape[1:]) != expected_shape:
            raise ValueError(
                "ACT proposal shape must match configured action15 chunk: "
                f"expected [B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(proposal_normalized.shape)}"
            )
        batch_size = proposal_normalized.shape[0]
        micro_proposals = proposal_normalized.reshape(
            batch_size,
            self.config.num_micro_chunks,
            self.config.micro_horizon,
            _LEFT15_ACTION_DIM,
        )
        endpoint_normalized, initial_velocity_normalized = self.knot_decoder(micro_proposals)
        endpoint_raw = self._denormalize_action(endpoint_normalized)
        # Keep the initial desired velocity visual-only as well.  It is the
        # missing velocity component of state8, not a measured dq or a reaction
        # state; the direct first-knot label supervises it during training.
        initial_action_normalized = proposal_normalized[:, 0].clone()
        initial_action_normalized[:, _ARM7_DIM : 2 * _ARM7_DIM] = initial_velocity_normalized
        initial_action_raw = self._denormalize_action(initial_action_normalized)
        initial_velocity_raw = initial_action_raw[:, _ARM7_DIM : 2 * _ARM7_DIM]
        if self.config.max_joint_velocity is not None:
            initial_velocity_raw = initial_velocity_raw.clamp(
                min=-float(self.config.max_joint_velocity),
                max=float(self.config.max_joint_velocity),
            )
            initial_action_raw = initial_action_raw.clone()
            initial_action_raw[:, _ARM7_DIM : 2 * _ARM7_DIM] = initial_velocity_raw
        current_state_raw = self._current_state_raw(batch, reference=proposal_normalized)
        start_position = current_state_raw[:, :_ARM7_DIM]
        start_velocity = initial_velocity_raw
        start_gripper = current_state_raw[:, _ARM7_DIM : _ARM7_DIM + 1]

        segments: list[Tensor] = []
        start_positions: list[Tensor] = []
        start_velocities: list[Tensor] = []
        resolved_endpoints: list[Tensor] = []
        interpolation = (
            torch.arange(
                1,
                self.config.micro_horizon + 1,
                dtype=proposal_normalized.dtype,
                device=proposal_normalized.device,
            )
            / float(self.config.micro_horizon)
        ).view(1, self.config.micro_horizon, 1)
        for index in range(self.config.num_micro_chunks):
            endpoint = self._apply_motion_limits(start_position, endpoint_raw[:, index])
            start_positions.append(start_position)
            start_velocities.append(start_velocity)
            trajectory = sample_arm7_cubic_hermite(
                start_position,
                start_velocity,
                endpoint[:, :_ARM7_DIM],
                endpoint[:, _ARM7_DIM : 2 * _ARM7_DIM],
                duration=float(self.config.micro_horizon) * float(self.config.control_dt_s),
                num_samples=self.config.micro_horizon,
                include_start=False,
            )
            gripper = start_gripper.unsqueeze(1) + interpolation * (
                endpoint[:, 2 * _ARM7_DIM : 2 * _ARM7_DIM + 1].unsqueeze(1)
                - start_gripper.unsqueeze(1)
            )
            segments.append(torch.cat((trajectory.positions, trajectory.velocities, gripper), dim=-1))
            resolved_endpoints.append(endpoint)
            # This is knot chaining inside one generated curve, not an
            # environment transition or reaction-model feedback path.
            start_position = endpoint[:, :_ARM7_DIM]
            start_velocity = endpoint[:, _ARM7_DIM : 2 * _ARM7_DIM]
            start_gripper = endpoint[:, 2 * _ARM7_DIM : 2 * _ARM7_DIM + 1]

        resolved_endpoint_tensor = torch.stack(resolved_endpoints, dim=1)
        self._last_spline_endpoints_raw = resolved_endpoint_tensor.detach()
        self._last_spline_start_positions_raw = torch.stack(start_positions, dim=1).detach()
        self._last_spline_start_velocities_raw = torch.stack(start_velocities, dim=1).detach()
        actions_raw = torch.cat(segments, dim=1)
        # When an explicit safety limit is active, train the direct knot loss
        # against the same emitted endpoint that BC sees rather than the
        # unclamped decoder proposal.
        resolved_endpoints_normalized = self._normalize_action(resolved_endpoint_tensor)
        resolved_initial_velocity_normalized = self._normalize_action(initial_action_raw)[
            :, _ARM7_DIM : 2 * _ARM7_DIM
        ]
        return (
            self._normalize_action(actions_raw),
            resolved_endpoints_normalized,
            resolved_initial_velocity_normalized,
        )

    def _smoothness_losses(self, normalized_actions: Tensor) -> tuple[Tensor, Tensor]:
        raw = self._denormalize_action(normalized_actions)
        if raw.shape[1] < 2:
            zero = raw.new_zeros(())
            return zero, zero
        dt = float(self.config.control_dt_s)
        position = raw[:, :, :_ARM7_DIM]
        velocity = raw[:, :, _ARM7_DIM : 2 * _ARM7_DIM]
        integration = (
            position[:, 1:] - position[:, :-1] - 0.5 * dt * (velocity[:, 1:] + velocity[:, :-1])
        ).square().mean()
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        smoothness = acceleration.square().mean()
        if acceleration.shape[1] >= 2:
            jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
            smoothness = smoothness + jerk.square().mean()
        return integration, smoothness

    def _validate_training_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        actions = batch.get(ACTION)
        action_is_pad = batch.get("action_is_pad")
        expected_shape = (self.config.chunk_size, _LEFT15_ACTION_DIM)
        if not isinstance(actions, Tensor) or actions.ndim != 3 or tuple(actions.shape[1:]) != expected_shape:
            actual = tuple(actions.shape) if isinstance(actions, Tensor) else None
            raise ValueError(
                "Spline ACT training action must be "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], got {actual}"
            )
        if (
            not isinstance(action_is_pad, Tensor)
            or action_is_pad.shape != actions.shape[:2]
            or action_is_pad.dtype != torch.bool
        ):
            actual_shape = tuple(action_is_pad.shape) if isinstance(action_is_pad, Tensor) else None
            actual_dtype = action_is_pad.dtype if isinstance(action_is_pad, Tensor) else None
            raise ValueError(
                "Spline ACT action_is_pad must be bool [B, chunk_size], "
                f"got shape={actual_shape}, dtype={actual_dtype}"
            )
        return actions, action_is_pad

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        proposal, _ = self.model(self._prepare_model_batch(batch))
        actions, _, _ = self._decode_spline_action_chunk(batch, proposal)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        target_actions, target_is_pad = self._validate_training_targets(batch)
        proposal, (mu_hat, log_sigma_x2_hat) = self.model(self._prepare_model_batch(batch))
        actions_hat, endpoint_normalized, initial_velocity_normalized = self._decode_spline_action_chunk(
            batch, proposal
        )
        l1_loss = (
            F.l1_loss(target_actions, actions_hat, reduction="none") * ~target_is_pad.unsqueeze(-1)
        ).mean()
        endpoint_indices = torch.arange(
            self.config.micro_horizon - 1,
            self.config.chunk_size,
            self.config.micro_horizon,
            device=target_actions.device,
        )
        endpoint_targets = target_actions.index_select(1, endpoint_indices)
        endpoint_is_pad = target_is_pad.index_select(1, endpoint_indices)
        endpoint_loss = (
            F.l1_loss(endpoint_normalized, endpoint_targets, reduction="none")
            * ~endpoint_is_pad.unsqueeze(-1)
        ).mean()
        initial_velocity_loss = (
            F.l1_loss(
                initial_velocity_normalized,
                target_actions[:, 0, _ARM7_DIM : 2 * _ARM7_DIM],
                reduction="none",
            )
            * ~target_is_pad[:, 0].unsqueeze(-1)
        ).mean()
        knot_loss = endpoint_loss + initial_velocity_loss
        integration_loss, smoothness_loss = self._smoothness_losses(actions_hat)
        loss = self.config.action_bc_weight * l1_loss
        loss = loss + self.config.knot_loss_weight * knot_loss
        loss = loss + self.config.integration_weight * integration_loss
        loss = loss + self.config.smoothness_weight * smoothness_loss
        loss_dict: dict[str, float] = {
            "l1_loss": l1_loss.item(),
            "knot_loss": knot_loss.item(),
            "endpoint_loss": endpoint_loss.item(),
            "initial_velocity_loss": initial_velocity_loss.item(),
            "integration_loss": integration_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
        }
        if self.config.use_vae:
            if mu_hat is None or log_sigma_x2_hat is None:
                raise AssertionError("ACT VAE must return latent parameters during training")
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight
        return loss, loss_dict


__all__ = ["SplineACTPolicy", "SplineKnotDecoder"]
