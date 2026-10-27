"""Configuration for the out-of-tree Reaction-Aware ACT policy.

The policy deliberately subclasses LeRobot's :class:`ACTConfig` instead of
modifying upstream ACT.  With ``reaction_aware=False`` it is an exact ACT
execution path; enabling the flag opts into the explicitly causal, latest-plan
rollout path.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .reaction_phase_gate import (
    PRECONTACT_CONTROL_STATES,
    PRECONTACT_CONTROL_STATE_SEQUENCE,
    normalize_precontact_control_states,
)
from .reaction_state import REACTION_MODEL_STATE_DIM


@PreTrainedConfig.register_subclass("reaction_aware_act")
@dataclass
class ReactionAwareACTConfig(ACTConfig):
    """ACT settings plus the bounded first implementation of Reaction rollout.

    ``reaction_aware`` is off by default so the plugin remains a useful
    checkpoint-compatible ACT control path.  The enabled configuration enforces
    replan-and-execute-one-step semantics and partitions ``chunk_size`` into
    fixed-size Hermite micro-chunks.
    """

    # Preserve the native ACT path unless explicitly enabled.
    reaction_aware: bool = False

    # Causal rollout and action representation.
    action_representation: str = "original"
    micro_horizon: int = 4
    num_micro_chunks: int = 8
    replan_every_step: bool = True
    use_latest_chunk_only: bool = True
    predict_joint_velocity: bool = True
    control_dt_s: float = 0.02
    max_joint_delta: float | None = None
    max_joint_velocity: float | None = None

    # Reaction backend.  Only learned_transition is implemented in phase one;
    # the explicit field prevents callers from assuming it is an analytical GJM.
    reaction_model_enabled: bool = True
    reaction_model_type: str = "learned_transition"
    reaction_model_checkpoint: str | None = None
    freeze_reaction_model: bool = True
    allow_untrained_reaction_model_for_smoke: bool = False
    reaction_hidden_dim: int = 256
    reaction_num_layers: int = 3
    reaction_predict_uncertainty: bool = False

    # Runtime contract keys.  They are raw physical quantities, intentionally
    # excluded from ACT's regular normalized observation vector.
    reaction_state_key: str = "observation.reaction_state"
    reaction_state_dim: int = 80
    # V1 R0 is only valid on the canonical pre-contact state-machine support.
    # This is an admission contract for the server/processor, never an ACT or
    # reaction-model feature.  There is deliberately no "disabled" path while
    # reaction_aware=True because no contact-phase fallback exists yet.
    require_precontact_control_state: bool = True
    precontact_control_states: tuple[int, ...] = PRECONTACT_CONTROL_STATE_SEQUENCE
    # V1 policy keeps the existing RGB ACT condition.  Canonical relative target
    # pose/EE diagnostics are simulator-derived audit labels, not a privileged
    # policy input in the first visual-only ablation.
    task_state_mode: str = "visual_latent"

    # Auxiliary losses.  Defaults make the first policy training stage pure BC
    # with a frozen reaction model; callers must opt in to extra objectives.
    action_bc_weight: float = 1.0
    integration_weight: float = 0.0
    smoothness_weight: float = 0.0
    reaction_pose_weight: float = 1.0
    reaction_twist_weight: float = 1.0
    rollout_weight: float = 0.0
    terminal_weight: float = 0.0

    # B4 composition: B3's action-conditioned, train-label-only base head is
    # optional at the config-class level solely so narrow legacy/core-rollout
    # tests remain possible.  The canonical B4 profile, launcher and server
    # require it to be enabled.  Its prediction is never a reaction-model or
    # action-decoder input.
    base_auxiliary_enabled: bool = False
    base_auxiliary_horizon: int = 4
    base_auxiliary_hidden_dim: int = 256
    base_auxiliary_loss_weight: float = 1.0
    base_auxiliary_translation_weight: float = 1.0
    base_auxiliary_rotation_weight: float = 1.0
    base_auxiliary_twist_weight: float = 1.0
    base_auxiliary_label_schema: str = "body_delta_se3_future_body_twist_v1"

    # Dataset-contract assertions.  ``None`` preserves generic test fixtures
    # and legacy ACT use; phase-one canonical runs set these explicitly.
    expected_state_dim: int | None = None
    expected_action_dim: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_reaction_hyperparameters()

    def _validate_reaction_hyperparameters(self) -> None:
        if self.action_representation not in {"original", "hermite_joint_trajectory"}:
            raise ValueError(
                "action_representation must be 'original' or 'hermite_joint_trajectory'"
            )
        if self.reaction_model_type not in {
            "learned_transition",
            "mechanical_connection",
            "analytical_gjm",
            "gjm_residual",
        }:
            raise ValueError(f"unsupported reaction_model_type={self.reaction_model_type!r}")
        if self.task_state_mode != "visual_latent":
            raise ValueError(
                "phase-one Reaction-Aware ACT permits only task_state_mode='visual_latent'; "
                "canonical target/EE diagnostics are audit-only and must not become policy inputs"
            )
        if self.micro_horizon <= 0 or self.num_micro_chunks <= 0:
            raise ValueError("micro_horizon and num_micro_chunks must be positive")
        if self.reaction_state_dim <= 0:
            raise ValueError("reaction_state_dim must be positive")
        if self.reaction_hidden_dim <= 0 or self.reaction_num_layers <= 0:
            raise ValueError("reaction_hidden_dim and reaction_num_layers must be positive")
        if self.max_joint_delta is not None and self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive when set")
        if self.max_joint_velocity is not None and self.max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive when set")
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        loss_weights = {
            "action_bc_weight": self.action_bc_weight,
            "integration_weight": self.integration_weight,
            "smoothness_weight": self.smoothness_weight,
            "reaction_pose_weight": self.reaction_pose_weight,
            "reaction_twist_weight": self.reaction_twist_weight,
            "rollout_weight": self.rollout_weight,
            "terminal_weight": self.terminal_weight,
            "base_auxiliary_loss_weight": self.base_auxiliary_loss_weight,
            "base_auxiliary_translation_weight": self.base_auxiliary_translation_weight,
            "base_auxiliary_rotation_weight": self.base_auxiliary_rotation_weight,
            "base_auxiliary_twist_weight": self.base_auxiliary_twist_weight,
        }
        if any(value < 0.0 for value in loss_weights.values()):
            raise ValueError(f"reaction loss weights must be non-negative, got {loss_weights}")

        precontact_states = normalize_precontact_control_states(
            self.precontact_control_states,
            name="precontact_control_states",
        )
        self.precontact_control_states = precontact_states

        if not self.reaction_aware:
            if self.base_auxiliary_enabled:
                raise ValueError(
                    "base_auxiliary_enabled belongs to the B4 reaction-aware composition; "
                    "use base_aux_spline_act for B3"
                )
            return
        if self.require_precontact_control_state is not True:
            raise ValueError(
                "reaction_aware=True requires require_precontact_control_state=true; "
                "v1 has no post-contact fallback"
            )
        if frozenset(precontact_states) != PRECONTACT_CONTROL_STATES:
            raise ValueError(
                "reaction_aware=True requires canonical precontact_control_states="
                f"{list(PRECONTACT_CONTROL_STATE_SEQUENCE)}; v1 has no contact fallback"
            )
        if self.action_representation != "hermite_joint_trajectory":
            raise ValueError("reaction_aware=True requires action_representation='hermite_joint_trajectory'")
        if not self.reaction_model_enabled:
            raise ValueError("reaction_aware=True requires reaction_model_enabled=True")
        if self.reaction_model_type != "learned_transition":
            raise NotImplementedError(
                "phase-one Reaction-Aware ACT implements only reaction_model_type='learned_transition'"
            )
        if self.n_action_steps != 1:
            raise ValueError("reaction_aware=True requires n_action_steps=1 for latest-plan execution")
        if self.temporal_ensemble_coeff is not None:
            raise ValueError("reaction_aware=True is incompatible with ACT temporal ensembling")
        if not self.replan_every_step or not self.use_latest_chunk_only:
            raise ValueError("reaction_aware=True requires replan_every_step and use_latest_chunk_only")
        if self.num_micro_chunks < 2:
            raise ValueError("reaction_aware=True requires at least two causal micro-chunks")
        if self.reaction_state_dim != REACTION_MODEL_STATE_DIM:
            raise ValueError(
                "phase-one Reaction-Aware ACT requires the fixed 80D raw reaction state; "
                f"got reaction_state_dim={self.reaction_state_dim}"
            )
        if not self.predict_joint_velocity:
            raise ValueError(
                "reaction_aware=True requires predict_joint_velocity=True for the canonical "
                "Left15 position/desired-velocity action contract"
            )
        if self.micro_horizon * self.num_micro_chunks != self.chunk_size:
            raise ValueError(
                "reaction-aware chunk_size must equal micro_horizon * num_micro_chunks"
            )
        if self.rollout_weight != 0.0 or self.terminal_weight != 0.0:
            raise ValueError(
                "phase-one policy training has no future-state labels in the ACT batch; "
                "keep rollout_weight=terminal_weight=0 and train reaction supervision separately"
            )
        if self.base_auxiliary_enabled:
            if self.base_auxiliary_horizon <= 0:
                raise ValueError("base_auxiliary_horizon must be positive")
            if self.base_auxiliary_horizon != self.micro_horizon:
                raise ValueError(
                    "B4 base_auxiliary_horizon must equal micro_horizon so the head "
                    "uses exactly the first emitted reaction micro-chunk"
                )
            if self.base_auxiliary_hidden_dim <= 0:
                raise ValueError("base_auxiliary_hidden_dim must be positive")
            if self.base_auxiliary_loss_weight <= 0.0:
                raise ValueError("B4 base_auxiliary_loss_weight must be positive when enabled")
            if (
                self.base_auxiliary_translation_weight
                + self.base_auxiliary_rotation_weight
                + self.base_auxiliary_twist_weight
                <= 0.0
            ):
                raise ValueError("B4 base auxiliary needs at least one positive target loss weight")
            if self.base_auxiliary_label_schema != "body_delta_se3_future_body_twist_v1":
                raise ValueError(
                    "B4 only supports base_auxiliary_label_schema="
                    "'body_delta_se3_future_body_twist_v1'"
                )
            if not self.freeze_reaction_model:
                raise ValueError(
                    "B4 base auxiliary composition requires freeze_reaction_model=True; "
                    "R0 is never fine-tuned in policy training"
                )

    def validate_features(self) -> None:
        super().validate_features()
        if self.robot_state_feature is None:
            raise ValueError("Dataset must provide observation.state")
        if self.action_feature is None:
            raise ValueError("Dataset must provide action")
        if self.expected_state_dim is not None and self.robot_state_feature.shape != (
            self.expected_state_dim,
        ):
            raise ValueError(
                f"Expected observation.state shape {(self.expected_state_dim,)}, "
                f"got {self.robot_state_feature.shape}"
            )
        if self.expected_action_dim is not None and self.action_feature.shape != (
            self.expected_action_dim,
        ):
            raise ValueError(
                f"Expected action shape {(self.expected_action_dim,)}, got {self.action_feature.shape}"
            )
        if not self.reaction_aware:
            return
        if self.robot_state_feature.shape != (8,):
            raise ValueError(
                "phase-one Reaction-Aware ACT requires 8D left policy state; "
                f"got {self.robot_state_feature.shape}"
            )
        if not self.image_features:
            raise ValueError(
                "phase-one Reaction-Aware ACT keeps the visual ACT condition and requires at least one image"
            )
        if self.action_feature.shape != (15,):
            raise ValueError(
                "phase-one Reaction-Aware ACT requires action15: arm position7 + "
                f"desired velocity7 + gripper position1; got {self.action_feature.shape}"
            )

    @property
    def action_delta_indices(self) -> list:
        """Request the causal ``a[t-1] + a[t:t+H-1]`` training window.

        Native ACT supervises exactly ``H`` future actions.  The enabled
        reaction path additionally needs the actually preceding command as a
        physical-state input, so its dataset window is one item longer while
        the ACT proposal and BC target remain exactly ``chunk_size`` long.
        At an episode boundary LeRobot marks the leading, clamped value in
        ``action_is_pad``; :class:`ReactionAwareACTPolicy` converts that single
        history item to the explicit all-zero cold-start command.
        """

        if self.reaction_aware:
            return [-1, *range(self.chunk_size)]
        return super().action_delta_indices

    @property
    def action_feature_name(self) -> str:
        """Make the expected action key explicit for diagnostics and tests."""
        return ACTION

    @property
    def robot_state_feature_name(self) -> str:
        """Make the expected policy-state key explicit for diagnostics and tests."""
        return OBS_STATE
