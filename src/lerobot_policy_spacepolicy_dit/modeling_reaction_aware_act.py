"""Reaction-Aware ACT implemented as an out-of-tree extension of native ACT.

The upstream ACT network is retained as the visual action-proposal backbone.
When enabled, this policy turns its parallel proposal into Hermite micro-chunks,
rolls each through a separately trained action-conditioned reaction model, and
uses the predicted physical state for the next micro-chunk.  The reaction model
never receives RGB, target pose, EE pose, or future ground-truth state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .base_auxiliary import (
    BASE_AUXILIARY_DIAGNOSTIC_KEYS,
    ActionConditionedBaseAuxiliaryHead,
    BaseAuxiliaryLoss,
    BaseAuxiliaryPrediction,
    base_auxiliary_loss,
    derive_base_auxiliary_labels,
)
from .configuration_reaction_aware_act import ReactionAwareACTConfig
from .reaction_model import (
    JOINT_POSITION_SLICE,
    JOINT_VELOCITY_SLICE,
    LEFT15_ACTION_DIM,
    LEFT_ARM_INDICES,
    REACTION_STATE_DIM,
    LearnedReactionModel,
    ReactionPrediction,
)
from .trajectory import sample_arm7_cubic_hermite


_DIAGNOSTIC_BASE_POSE = "diagnostics.base_pose"
_DIAGNOSTIC_BASE_VELOCITY = "diagnostics.base_velocity"
_DIAGNOSTIC_RAW_JOINT_STATE = "diagnostics.raw_joint_state"
_DIAGNOSTIC_MEASURED_JOINT_VELOCITY = "diagnostics.measured_joint_velocity"
_REACTION_MODEL_CHECKPOINT_SCHEMA_VERSION = "reaction_model_checkpoint_v1"


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


class CausalMicroChunkDecoder(nn.Module):
    """Decode one terminal Hermite knot from visual ACT proposal + predicted state.

    The ACT proposal has already encoded RGB and robot state.  This small decoder
    has no target/EE diagnostic input: canonical relative target pose is reserved
    for audit metrics in the v1 visual-only comparison.
    """

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if action_dim != LEFT15_ACTION_DIM:
            raise ValueError(f"phase-one micro decoder expects action15, got {action_dim}")
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2 * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Start close to the visual ACT proposal while preserving a trainable,
        # state-dependent causal correction path.
        last = self.network[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)

    def forward(self, robot_state: Tensor, proposal_micro_chunk: Tensor) -> Tensor:
        if robot_state.ndim != 2 or robot_state.shape[-1] != REACTION_STATE_DIM:
            raise ValueError(
                "robot_state must have shape [B, 80], "
                f"got {tuple(robot_state.shape)}"
            )
        if proposal_micro_chunk.ndim != 3 or proposal_micro_chunk.shape[-1] != LEFT15_ACTION_DIM:
            raise ValueError(
                "proposal_micro_chunk must have shape [B, H, 15], "
                f"got {tuple(proposal_micro_chunk.shape)}"
            )
        if robot_state.shape[0] != proposal_micro_chunk.shape[0]:
            raise ValueError("robot_state and proposal micro-chunk batch sizes must match")
        proposal_summary = torch.cat(
            (proposal_micro_chunk.mean(dim=1), proposal_micro_chunk[:, -1]), dim=-1
        )
        return self.network(torch.cat((robot_state, proposal_summary), dim=-1))


class ReactionAwareACTPolicy(ACTPolicy):
    """Native ACT with optional causal Reaction rollout.

    With ``config.reaction_aware=False`` all public calls delegate to upstream
    :class:`ACTPolicy`, retaining its action shape, queue behavior, VAE objective,
    and temporal-ensemble support.  Enabled mode is deliberately sealed to
    latest-plan execution (one action per call, no temporal ensemble).
    """

    config_class = ReactionAwareACTConfig
    name = "reaction_aware_act"

    def __init__(
        self,
        config: ReactionAwareACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ) -> None:
        # ACT ignores dataset_stats; this extension uses them only to convert its
        # raw Hermite path back to the policy processor's normalized action space.
        del kwargs
        self._reaction_dataset_stats = dataset_stats
        super().__init__(config)
        self.config = config
        self._register_action_normalization_buffers(dataset_stats)
        self.reaction_model: LearnedReactionModel | None = None
        self.micro_chunk_decoder: CausalMicroChunkDecoder | None = None
        self._last_reaction_predictions: tuple[ReactionPrediction, ...] = ()
        self.base_auxiliary_head: ActionConditionedBaseAuxiliaryHead | None = None
        self._last_base_auxiliary_prediction: BaseAuxiliaryPrediction | None = None
        self._last_base_auxiliary_loss: BaseAuxiliaryLoss | None = None

        if config.reaction_aware:
            self.reaction_model = LearnedReactionModel(
                state_dim=config.reaction_state_dim,
                action_dim=config.action_feature.shape[0],
                hidden_dim=config.reaction_hidden_dim,
                num_layers=config.reaction_num_layers,
                predict_uncertainty=config.reaction_predict_uncertainty,
            )
            if config.reaction_model_checkpoint is not None:
                self._load_reaction_model_checkpoint(config.reaction_model_checkpoint)
            self.micro_chunk_decoder = CausalMicroChunkDecoder(
                state_dim=config.reaction_state_dim,
                action_dim=config.action_feature.shape[0],
                hidden_dim=config.reaction_hidden_dim,
            )
            if config.base_auxiliary_enabled:
                self.base_auxiliary_head = ActionConditionedBaseAuxiliaryHead(
                    action_dim=config.action_feature.shape[0],
                    horizon=config.base_auxiliary_horizon,
                    hidden_dim=config.base_auxiliary_hidden_dim,
                )
            if config.freeze_reaction_model:
                for parameter in self.reaction_model.parameters():
                    parameter.requires_grad_(False)
                self.reaction_model.eval()

    def _register_action_normalization_buffers(
        self, dataset_stats: Mapping[str, Mapping[str, Any]] | None
    ) -> None:
        action_feature = self.config.action_feature
        if action_feature is None:
            # ``super().__init__`` has already validated this for actual policies;
            # retain an explicit guard for direct unit construction.
            raise ValueError("Reaction-Aware ACT requires an action feature")
        action_dim = action_feature.shape[0]
        self.register_buffer("reaction_action_mean", _stats_vector(dataset_stats, ACTION, "mean", action_dim, 0.0))
        self.register_buffer("reaction_action_std", _stats_vector(dataset_stats, ACTION, "std", action_dim, 1.0))
        self.register_buffer("reaction_action_min", _stats_vector(dataset_stats, ACTION, "min", action_dim, -1.0))
        self.register_buffer("reaction_action_max", _stats_vector(dataset_stats, ACTION, "max", action_dim, 1.0))
        if bool(torch.any(self.reaction_action_std <= 0.0)):
            raise ValueError("action std must be positive for Reaction-Aware ACT")
        if bool(torch.any(self.reaction_action_max < self.reaction_action_min)):
            raise ValueError("action max must be >= action min for Reaction-Aware ACT")
        mode = self.config.normalization_mapping.get("ACTION", "IDENTITY")
        self._reaction_action_normalization = _mode_name(mode)

    def _normalize_action(self, action: Tensor) -> Tensor:
        mode = self._reaction_action_normalization
        if mode == "MEAN_STD":
            return (action - self.reaction_action_mean) / self.reaction_action_std.clamp_min(1.0e-6)
        if mode == "MIN_MAX":
            denominator = (self.reaction_action_max - self.reaction_action_min).clamp_min(1.0e-6)
            return 2.0 * (action - self.reaction_action_min) / denominator - 1.0
        if mode == "IDENTITY":
            return action
        raise ValueError(
            "Reaction-Aware ACT supports ACTION normalization only in "
            f"{{MEAN_STD, MIN_MAX, IDENTITY}}, got {mode!r}"
        )

    def _denormalize_action(self, action: Tensor) -> Tensor:
        mode = self._reaction_action_normalization
        if mode == "MEAN_STD":
            return action * self.reaction_action_std + self.reaction_action_mean
        if mode == "MIN_MAX":
            return (action + 1.0) * (self.reaction_action_max - self.reaction_action_min) / 2.0 + self.reaction_action_min
        if mode == "IDENTITY":
            return action
        raise AssertionError(f"unreachable action normalization mode {mode!r}")

    def _load_reaction_model_checkpoint(self, checkpoint_value: str) -> None:
        if self.reaction_model is None:
            raise AssertionError("reaction model must exist before loading its checkpoint")
        path = Path(checkpoint_value).expanduser()
        if path.is_dir():
            path = path / "reaction_model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"reaction_model_checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("reaction model checkpoint must contain a state-dict mapping")
        # A production R0 checkpoint is a standalone, independently trained
        # transition model.  Its temporal meaning is part of its interface: a
        # model trained to advance four 50-Hz commands cannot silently be used
        # as a two- or eight-step micro-chunk transition model.  Keep accepting
        # a bare state dict for narrow backwards-compatible development use, but
        # validate the complete schema whenever checkpoint metadata is present.
        has_metadata = any(
            key in payload
            for key in ("schema_version", "model_config", "transition_config", "train_split")
        )
        if has_metadata:
            if payload.get("schema_version") != _REACTION_MODEL_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    "reaction model checkpoint must use schema "
                    f"{_REACTION_MODEL_CHECKPOINT_SCHEMA_VERSION!r}"
                )
            model_config = payload.get("model_config")
            if not isinstance(model_config, Mapping):
                raise ValueError("reaction model checkpoint is missing model_config")
            expected_model_config = {
                "state_dim": self.config.reaction_state_dim,
                "action_dim": LEFT15_ACTION_DIM,
                "hidden_dim": self.config.reaction_hidden_dim,
                "num_layers": self.config.reaction_num_layers,
                "predict_uncertainty": self.config.reaction_predict_uncertainty,
            }
            incompatible = {
                key: (model_config.get(key), expected)
                for key, expected in expected_model_config.items()
                if model_config.get(key) != expected
            }
            if incompatible:
                raise ValueError(
                    "reaction model checkpoint configuration is incompatible with policy: "
                    f"{incompatible}"
                )
            transition_config = payload.get("transition_config")
            if not isinstance(transition_config, Mapping):
                raise ValueError("reaction model checkpoint is missing transition_config")
            if transition_config.get("candidate_steps") != self.config.micro_horizon:
                raise ValueError(
                    "reaction model checkpoint candidate_steps must match policy micro_horizon: "
                    f"{transition_config.get('candidate_steps')} vs {self.config.micro_horizon}"
                )
            if transition_config.get("future_offset_steps") != self.config.micro_horizon:
                raise ValueError(
                    "reaction model checkpoint future_offset_steps must equal policy micro_horizon: "
                    f"{transition_config.get('future_offset_steps')} vs {self.config.micro_horizon}"
                )
            checkpoint_dt = transition_config.get("expected_dt_s")
            if checkpoint_dt is not None and abs(float(checkpoint_dt) - self.config.control_dt_s) > 1.0e-8:
                raise ValueError(
                    "reaction model checkpoint expected_dt_s must match policy control_dt_s: "
                    f"{checkpoint_dt} vs {self.config.control_dt_s}"
                )
        state_dict = payload.get("reaction_model", payload.get("state_dict", payload))
        if not isinstance(state_dict, Mapping):
            raise ValueError("reaction model checkpoint has no mapping state_dict")
        # Permit a checkpoint saved from a containing policy without accepting an
        # arbitrary unrelated module namespace.
        if any(str(key).startswith("reaction_model.") for key in state_dict):
            state_dict = {
                str(key).removeprefix("reaction_model."): value
                for key, value in state_dict.items()
                if str(key).startswith("reaction_model.")
            }
        try:
            result = self.reaction_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError("reaction model checkpoint does not match current architecture") from error
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "reaction model checkpoint does not match current architecture: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save a self-contained policy rather than an absolute R0 dependency.

        ``reaction_model`` is part of this policy's state dict.  Persisting an
        external ``reaction_model_checkpoint`` path in ``config.json`` would make
        ``from_pretrained`` fail before those bundled weights are restored when
        the original R0 directory is moved or unavailable.  Clear the load-only
        path just while serializing configuration; the live policy retains it for
        provenance during the running job.
        """

        external_checkpoint = self.config.reaction_model_checkpoint
        if not self.config.reaction_aware or external_checkpoint is None:
            super()._save_pretrained(save_directory)
            return
        self.config.reaction_model_checkpoint = None
        try:
            super()._save_pretrained(save_directory)
        finally:
            self.config.reaction_model_checkpoint = external_checkpoint

    @staticmethod
    def _as_current_vector(value: Any, *, key: str, batch_size: int, width: int, reference: Tensor) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        elif (
            tensor.ndim == 3
            and tensor.shape[1] == 2
            and key in {_DIAGNOSTIC_BASE_POSE, _DIAGNOSTIC_BASE_VELOCITY}
        ):
            # B4's train-only base label window is [t, t+K].  R0 always sees
            # only t; selecting it here prevents future-label leakage even if
            # a caller hands the same retained diagnostic tensor to both paths.
            tensor = tensor[:, 0]
        if tensor.shape != (batch_size, width) or not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{key} must be finite with shape ({batch_size}, {width}), got {tuple(tensor.shape)}")
        return tensor

    def _reaction_state_from_batch(
        self,
        batch: dict[str, Tensor],
        *,
        batch_size: int,
        reference: Tensor,
        previous_command_raw: Tensor | None = None,
    ) -> Tensor:
        explicit = batch.get(self.config.reaction_state_key)
        if explicit is not None:
            return self._as_current_vector(
                explicit,
                key=self.config.reaction_state_key,
                batch_size=batch_size,
                width=REACTION_STATE_DIM,
                reference=reference,
            )

        required = (
            _DIAGNOSTIC_BASE_POSE,
            _DIAGNOSTIC_BASE_VELOCITY,
            _DIAGNOSTIC_RAW_JOINT_STATE,
            _DIAGNOSTIC_MEASURED_JOINT_VELOCITY,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "Reaction-Aware ACT needs raw reaction state online or canonical diagnostics during training; "
                f"missing={missing}"
            )
        # Offline policy training supplies the command at t-1 through the
        # explicitly extended action window. Inference uses the server-packed
        # state above. The zero fallback is only for direct diagnostic/smoke
        # calls without either source; it is never used by production training.
        if previous_command_raw is None:
            previous_command = torch.zeros(
                batch_size,
                LEFT15_ACTION_DIM,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            previous_command = self._as_current_vector(
                previous_command_raw,
                key="previous_left15_command",
                batch_size=batch_size,
                width=LEFT15_ACTION_DIM,
                reference=reference,
            )
        return torch.cat(
            (
                self._as_current_vector(batch[_DIAGNOSTIC_BASE_POSE], key=_DIAGNOSTIC_BASE_POSE, batch_size=batch_size, width=7, reference=reference),
                self._as_current_vector(batch[_DIAGNOSTIC_BASE_VELOCITY], key=_DIAGNOSTIC_BASE_VELOCITY, batch_size=batch_size, width=6, reference=reference),
                self._as_current_vector(batch[_DIAGNOSTIC_RAW_JOINT_STATE], key=_DIAGNOSTIC_RAW_JOINT_STATE, batch_size=batch_size, width=26, reference=reference),
                self._as_current_vector(batch[_DIAGNOSTIC_MEASURED_JOINT_VELOCITY], key=_DIAGNOSTIC_MEASURED_JOINT_VELOCITY, batch_size=batch_size, width=26, reference=reference),
                previous_command,
            ),
            dim=-1,
        )

    def _prepare_model_batch(
        self,
        batch: dict[str, Tensor],
        *,
        action_targets: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Do not rely on ACT silently ignoring extra diagnostics.  In enabled
        # B4 mode this exact whitelist is the policy input boundary: visual
        # observations plus canonical state8 and (only while training) the
        # original ACT action targets.  Raw reaction state and future base
        # labels are consumed by their dedicated paths below, never by ACT.
        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise KeyError("Reaction-Aware ACT requires observation.state")
        model_batch: dict[str, Tensor] = {OBS_STATE: state}
        if self.config.image_features:
            images: list[Tensor] = []
            for key in self.config.image_features:
                image = batch.get(key)
                if not isinstance(image, Tensor):
                    raise KeyError(f"Reaction-Aware ACT requires visual input {key!r}")
                images.append(image)
            model_batch[OBS_IMAGES] = images  # type: ignore[assignment]
        if action_targets is not None:
            model_batch[ACTION] = action_targets
        if action_is_pad is not None:
            model_batch["action_is_pad"] = action_is_pad
        return model_batch

    def _split_reaction_training_action_window(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Separate raw ``a[t-1]`` history from normalized ACT supervision.

        ``ReactionAwareACTConfig.action_delta_indices`` requests
        ``[-1, 0, ..., H-1]`` only when the reaction path is enabled. The
        leading action belongs to the physical reaction state; the remaining H
        normalized actions are the unmodified ACT VAE/BC target. A padded
        leading element is the episode cold-start convention, not a clamped
        copy of ``a[t]``.
        """

        actions = batch.get(ACTION)
        padding = batch.get("action_is_pad")
        if not isinstance(actions, Tensor) or not isinstance(padding, Tensor):
            raise KeyError(
                "reaction-aware training requires action and action_is_pad from the "
                "[-1, 0, ..., H-1] action window"
            )
        expected_shape = (self.config.chunk_size + 1, LEFT15_ACTION_DIM)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected_shape:
            raise ValueError(
                "reaction-aware training action must include one preceding command: "
                f"expected [B, {expected_shape[0]}, {expected_shape[1]}], got {tuple(actions.shape)}"
            )
        if padding.shape != actions.shape[:2] or padding.dtype is not torch.bool:
            raise ValueError(
                "reaction-aware training action_is_pad must be bool shape "
                f"{tuple(actions.shape[:2])}, got shape={tuple(padding.shape)}, dtype={padding.dtype}"
            )
        previous_raw = self._denormalize_action(actions[:, 0])
        previous_raw = torch.where(
            padding[:, 0].unsqueeze(-1),
            torch.zeros_like(previous_raw),
            previous_raw,
        )
        return previous_raw, actions[:, 1:], padding[:, 1:]

    def _base_auxiliary_labels(self, batch: dict[str, Tensor], *, reference: Tensor):
        missing = [key for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS if key not in batch]
        if missing:
            raise KeyError(
                "sealed B4 training requires train-label diagnostic windows "
                "diagnostics.base_pose and diagnostics.base_velocity; "
                f"missing={missing}"
            )
        return derive_base_auxiliary_labels(
            batch[_DIAGNOSTIC_BASE_POSE],
            batch[_DIAGNOSTIC_BASE_VELOCITY],
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
        """Return B3 supervision without any decoder/R0 output feedback.

        The first generated micro-chunk is exactly the K controls that the
        rollout will submit to R0.  Conditioning the head on it is therefore
        action-conditioned, while the head's prediction is never read by
        ``_make_micro_trajectory`` or ``reaction_model``.  This matches B3's
        train-only auxiliary semantics and does not create a second rollout.
        """

        if self.base_auxiliary_head is None:
            raise AssertionError("base auxiliary head must exist when B4 auxiliary is enabled")
        expected_shape = (self.config.chunk_size, LEFT15_ACTION_DIM)
        if emitted_actions_normalized.ndim != 3 or tuple(emitted_actions_normalized.shape[1:]) != expected_shape:
            raise ValueError(
                "B4 emitted action chunk must have canonical shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(emitted_actions_normalized.shape)}"
            )
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

    def _apply_motion_limits(self, current_state: Tensor, endpoint: Tensor) -> Tensor:
        result = endpoint.clone()
        arm_index = torch.tensor(LEFT_ARM_INDICES, device=current_state.device)
        current_position = current_state[:, JOINT_POSITION_SLICE][:, arm_index]
        if self.config.max_joint_delta is not None:
            limit = float(self.config.max_joint_delta)
            result[:, :7] = torch.maximum(
                torch.minimum(result[:, :7], current_position + limit), current_position - limit
            )
        if self.config.max_joint_velocity is not None:
            limit = float(self.config.max_joint_velocity)
            result[:, 7:14] = result[:, 7:14].clamp(min=-limit, max=limit)
        return result

    def _make_micro_trajectory(self, current_state: Tensor, proposal_micro_chunk_raw: Tensor) -> Tensor:
        if self.micro_chunk_decoder is None:
            raise AssertionError("micro decoder must exist in reaction-aware mode")
        correction = self.micro_chunk_decoder(current_state, proposal_micro_chunk_raw)
        # Conservative bounded correction scales keep initial rollout close to the
        # visual ACT proposal while still exposing an action/state causal path.
        correction_scale = torch.tensor(
            [0.10] * 7 + [0.50] * 7 + [0.05],
            dtype=proposal_micro_chunk_raw.dtype,
            device=proposal_micro_chunk_raw.device,
        )
        endpoint = proposal_micro_chunk_raw[:, -1] + torch.tanh(correction) * correction_scale
        endpoint = self._apply_motion_limits(current_state, endpoint)
        arm_index = torch.tensor(LEFT_ARM_INDICES, device=current_state.device)
        start_position = current_state[:, JOINT_POSITION_SLICE][:, arm_index]
        start_velocity = current_state[:, JOINT_VELOCITY_SLICE][:, arm_index]
        trajectory = sample_arm7_cubic_hermite(
            start_position,
            start_velocity,
            endpoint[:, :7],
            endpoint[:, 7:14],
            duration=float(self.config.micro_horizon) * float(self.config.control_dt_s),
            num_samples=self.config.micro_horizon,
            include_start=False,
        )
        gripper = endpoint[:, 14:15].unsqueeze(1).expand(-1, self.config.micro_horizon, -1)
        return torch.cat((trajectory.positions, trajectory.velocities, gripper), dim=-1)

    @staticmethod
    def _state_at_executed_micro_boundary(
        prediction: ReactionPrediction,
        candidate: Tensor,
    ) -> ReactionPrediction:
        """Use the generated Hermite end knot as the next planned arm boundary.

        The learned model may predict a tracking residual for independently
        supervised dynamics, but allowing that residual to become the next spline
        start would create a discontinuity between planned micro-chunks.  Planned
        arm ``q/dq`` therefore shares the exact generated endpoint; predicted base
        pose/twist and all passive-joint state remain reaction-model outputs.
        """

        next_state = prediction.next_robot_state.clone()
        arm_index = torch.tensor(LEFT_ARM_INDICES, device=next_state.device)
        final_action = candidate[:, -1]
        next_joint_position = next_state[:, JOINT_POSITION_SLICE]
        next_joint_velocity = next_state[:, JOINT_VELOCITY_SLICE]
        next_joint_position[:, arm_index] = final_action[:, :7]
        next_joint_velocity[:, arm_index] = final_action[:, 7:14]
        return ReactionPrediction(
            next_robot_state=next_state,
            next_base_pose_world_xyzw=prediction.next_base_pose_world_xyzw,
            next_base_twist_world=prediction.next_base_twist_world,
            body_delta_se3=prediction.body_delta_se3,
            uncertainty_log_variance=prediction.uncertainty_log_variance,
        )

    def _rollout_action_chunk(
        self,
        batch: dict[str, Tensor],
        proposal_normalized: Tensor,
        *,
        previous_command_raw: Tensor | None = None,
    ) -> Tensor:
        if self.reaction_model is None:
            raise AssertionError("reaction model must exist in reaction-aware mode")
        if proposal_normalized.shape[1:] != (self.config.chunk_size, LEFT15_ACTION_DIM):
            raise ValueError(
                "ACT proposal shape must match configured action15 chunk: "
                f"got {tuple(proposal_normalized.shape)}"
            )
        proposal_raw = self._denormalize_action(proposal_normalized)
        current_state = self._reaction_state_from_batch(
            batch,
            batch_size=proposal_raw.shape[0],
            reference=proposal_raw,
            previous_command_raw=previous_command_raw,
        )
        output_micro_chunks: list[Tensor] = []
        predictions: list[ReactionPrediction] = []
        if self.config.freeze_reaction_model:
            self.reaction_model.eval()
        for index in range(self.config.num_micro_chunks):
            start = index * self.config.micro_horizon
            end = start + self.config.micro_horizon
            candidate = self._make_micro_trajectory(current_state, proposal_raw[:, start:end])
            output_micro_chunks.append(candidate)
            prediction = self._state_at_executed_micro_boundary(
                self.reaction_model(current_state, candidate),
                candidate,
            )
            predictions.append(prediction)
            current_state = prediction.next_robot_state
        self._last_reaction_predictions = tuple(predictions)
        raw_actions = torch.cat(output_micro_chunks, dim=1)
        return self._normalize_action(raw_actions)

    def _smoothness_losses(self, normalized_actions: Tensor) -> tuple[Tensor, Tensor]:
        raw = self._denormalize_action(normalized_actions)
        dt = float(self.config.control_dt_s)
        position = raw[:, :, :7]
        velocity = raw[:, :, 7:14]
        if raw.shape[1] < 2:
            zero = raw.new_zeros(())
            return zero, zero
        integration = (
            position[:, 1:] - position[:, :-1] - 0.5 * dt * (velocity[:, 1:] + velocity[:, :-1])
        ).square().mean()
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        smoothness = acceleration.square().mean()
        if acceleration.shape[1] >= 2:
            jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
            smoothness = smoothness + jerk.square().mean()
        return integration, smoothness

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not self.config.reaction_aware:
            return super().predict_action_chunk(batch, **kwargs)
        if kwargs:
            raise TypeError(f"unexpected Reaction-Aware ACT predict kwargs: {sorted(kwargs)}")
        self.eval()
        proposal, _ = self.model(self._prepare_model_batch(batch))
        return self._rollout_action_chunk(batch, proposal)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if not self.config.reaction_aware:
            return super().forward(batch)
        previous_command_raw, target_actions, target_is_pad = self._split_reaction_training_action_window(batch)
        model_batch = self._prepare_model_batch(
            batch,
            action_targets=target_actions,
            action_is_pad=target_is_pad,
        )
        proposal, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)
        actions_hat = self._rollout_action_chunk(
            batch,
            proposal,
            previous_command_raw=previous_command_raw,
        )
        l1_loss = (
            F.l1_loss(target_actions, actions_hat, reduction="none") * ~target_is_pad.unsqueeze(-1)
        ).mean()
        integration_loss, smoothness_loss = self._smoothness_losses(actions_hat)
        loss = self.config.action_bc_weight * l1_loss
        loss = loss + self.config.integration_weight * integration_loss
        loss = loss + self.config.smoothness_weight * smoothness_loss
        loss_dict: dict[str, float] = {
            "l1_loss": l1_loss.item(),
            "integration_loss": integration_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
        }
        if self.config.base_auxiliary_enabled:
            auxiliary = self._base_auxiliary_terms(
                batch,
                emitted_actions_normalized=actions_hat,
            )
            loss = loss + self.config.base_auxiliary_loss_weight * auxiliary.total
            loss_dict.update(
                {
                    "base_auxiliary_loss": auxiliary.total.item(),
                    "base_auxiliary_translation_loss": auxiliary.translation.item(),
                    "base_auxiliary_rotation_loss": auxiliary.rotation.item(),
                    "base_auxiliary_twist_loss": auxiliary.twist.item(),
                    "base_auxiliary_valid_samples": float(auxiliary.valid_samples.item()),
                }
            )
        if self.config.use_vae:
            if mu_hat is None or log_sigma_x2_hat is None:
                raise AssertionError("ACT VAE must return latent parameters during training")
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight
        return loss, loss_dict


__all__ = ["CausalMicroChunkDecoder", "ReactionAwareACTPolicy"]
