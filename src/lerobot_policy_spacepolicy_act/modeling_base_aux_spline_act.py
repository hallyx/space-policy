"""B3 policy: B2 Hermite ACT plus an output-only action-conditioned base head.

There is a deliberately one-way graph in the inference direction:

``visual + state8 -> ACT/Hermite action15 -> B3 head (training only)``

The base prediction has no edge back into ``ACT/Hermite action15``.  B3 does
not construct a ReactionState, load R0, or roll a predicted physical state into
a later micro-chunk.  During optimization the auxiliary loss can of course
shape the shared action proposal through ordinary backpropagation; that is the
point of an auxiliary-supervision ablation, not a runtime feedback path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .base_auxiliary import (
    BASE_AUXILIARY_DIAGNOSTIC_KEYS,
    ActionConditionedBaseAuxiliaryHead,
    BaseAuxiliaryLoss,
    BaseAuxiliaryPrediction,
    base_auxiliary_loss,
    derive_base_auxiliary_labels,
)
from .configuration_base_aux_spline_act import BaseAuxiliarySplineACTConfig
from .modeling_spline_act import SplineACTPolicy


_LEFT15_ACTION_DIM = 15
_ARM7_DIM = 7


class BaseAuxiliarySplineACTPolicy(SplineACTPolicy):
    """Independent B3 type with a strict label-only diagnostic boundary."""

    config_class = BaseAuxiliarySplineACTConfig
    name = "base_aux_spline_act"

    def __init__(
        self,
        config: BaseAuxiliarySplineACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, dataset_stats=dataset_stats, **kwargs)
        self.config = config
        self.base_auxiliary_head = ActionConditionedBaseAuxiliaryHead(
            action_dim=config.action_feature.shape[0],
            horizon=config.base_auxiliary_horizon,
            hidden_dim=config.base_auxiliary_hidden_dim,
        )
        # Audit-only training trace.  It is intentionally never queried by
        # predict_action_chunk/select_action and therefore cannot become an
        # accidental online input or decoder feedback channel.
        self._last_base_auxiliary_prediction: BaseAuxiliaryPrediction | None = None
        self._last_base_auxiliary_loss: BaseAuxiliaryLoss | None = None

    def _prepare_model_batch(
        self,
        batch: dict[str, Tensor],
        *,
        action_targets: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Build the *only* input dictionary supplied to the ACT backbone.

        The raw B3 diagnostic windows are retained by a custom processor only
        for loss construction below.  They are explicitly omitted here, as are
        every target/EE/future field.  This whitelist makes the policy-input
        boundary testable rather than relying on ACT silently ignoring extras.
        """

        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise KeyError("B3 requires canonical observation.state")
        model_batch: dict[str, Tensor] = {OBS_STATE: state}
        if self.config.image_features:
            images: list[Tensor] = []
            for key in self.config.image_features:
                image = batch.get(key)
                if not isinstance(image, Tensor):
                    raise KeyError(f"B3 requires visual input {key!r}")
                images.append(image)
            model_batch[OBS_IMAGES] = images  # type: ignore[assignment]
        if action_targets is not None:
            model_batch[ACTION] = action_targets
        if action_is_pad is not None:
            model_batch["action_is_pad"] = action_is_pad
        return model_batch

    def _base_auxiliary_labels(self, batch: dict[str, Tensor], *, reference: Tensor):
        missing = [key for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS if key not in batch]
        if missing:
            raise KeyError(
                "B3 training requires only train-label diagnostic windows "
                "diagnostics.base_pose and diagnostics.base_velocity; "
                f"missing={missing}"
            )
        return derive_base_auxiliary_labels(
            batch["diagnostics.base_pose"],
            batch["diagnostics.base_velocity"],
            base_pose_is_pad=batch.get("diagnostics.base_pose_is_pad"),
            base_velocity_is_pad=batch.get("diagnostics.base_velocity_is_pad"),
            reference=reference,
        )

    def _base_auxiliary_terms(
        self,
        batch: dict[str, Tensor],
        *,
        emitted_actions_normalized: Tensor,
    ) -> BaseAuxiliaryLoss:
        if emitted_actions_normalized.shape[1:] != (self.config.chunk_size, _LEFT15_ACTION_DIM):
            raise ValueError(
                "B3 emitted action chunk must have canonical shape "
                f"[B, {self.config.chunk_size}, 15], got {tuple(emitted_actions_normalized.shape)}"
            )
        # The condition is the exact physical candidate that would be executed
        # by B2.  It contains no diagnostic base state and nothing predicted by
        # this head.  No head output is passed to _decode_spline_action_chunk.
        candidate_action_raw = self._denormalize_action(
            emitted_actions_normalized[:, : self.config.base_auxiliary_horizon]
        )
        prediction = self.base_auxiliary_head(candidate_action_raw)
        labels = self._base_auxiliary_labels(batch, reference=candidate_action_raw)
        losses = base_auxiliary_loss(
            prediction,
            labels,
            translation_weight=self.config.base_auxiliary_translation_weight,
            rotation_weight=self.config.base_auxiliary_rotation_weight,
            twist_weight=self.config.base_auxiliary_twist_weight,
        )
        self._last_base_auxiliary_prediction = BaseAuxiliaryPrediction(
            body_delta_se3=prediction.body_delta_se3.detach(),
            future_twist_body=prediction.future_twist_body.detach(),
        )
        self._last_base_auxiliary_loss = losses
        return losses

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute ordinary B2 losses plus a train-only B3 base objective."""

        target_actions, target_is_pad = self._validate_training_targets(batch)
        proposal, (mu_hat, log_sigma_x2_hat) = self.model(
            self._prepare_model_batch(
                batch,
                action_targets=target_actions,
                action_is_pad=target_is_pad,
            )
        )
        actions_hat, endpoint_normalized, initial_velocity_normalized = self._decode_spline_action_chunk(
            batch, proposal
        )
        l1_loss = (
            F.l1_loss(target_actions, actions_hat, reduction="none")
            * ~target_is_pad.unsqueeze(-1)
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
        auxiliary = self._base_auxiliary_terms(batch, emitted_actions_normalized=actions_hat)

        loss = self.config.action_bc_weight * l1_loss
        loss = loss + self.config.knot_loss_weight * knot_loss
        loss = loss + self.config.integration_weight * integration_loss
        loss = loss + self.config.smoothness_weight * smoothness_loss
        loss = loss + self.config.base_auxiliary_loss_weight * auxiliary.total
        loss_dict: dict[str, float] = {
            "l1_loss": l1_loss.item(),
            "knot_loss": knot_loss.item(),
            "endpoint_loss": endpoint_loss.item(),
            "initial_velocity_loss": initial_velocity_loss.item(),
            "integration_loss": integration_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
            "base_auxiliary_loss": auxiliary.total.item(),
            "base_auxiliary_translation_loss": auxiliary.translation.item(),
            "base_auxiliary_rotation_loss": auxiliary.rotation.item(),
            "base_auxiliary_twist_loss": auxiliary.twist.item(),
            "base_auxiliary_valid_samples": float(auxiliary.valid_samples.item()),
        }
        if self.config.use_vae:
            if mu_hat is None or log_sigma_x2_hat is None:
                raise AssertionError("ACT VAE must return latent parameters during training")
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight
        return loss, loss_dict


__all__ = ["BaseAuxiliarySplineACTPolicy"]
