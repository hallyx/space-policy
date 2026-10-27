"""Configuration for the B2, no-rollout Spline ACT ablation.

``SplineACTConfig`` is deliberately a separate third-party policy type rather
than a switch on ACT or Reaction-Aware ACT.  It preserves the ACT visual
proposal network and normal processor contract, then deterministically decodes
that proposal as piecewise cubic Hermite arm trajectories.  It does *not* own a
reaction state, transition model, base auxiliary head, or causal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("spline_act")
@dataclass
class SplineACTConfig(ACTConfig):
    """ACT with a sealed B2 Hermite action decoder.

    The B2 comparison changes exactly the action representation and execution
    contract relative to C0: visual ACT produces a chunk, and the chunk is
    decoded through fixed-horizon cubic Hermite segments.  There is no learned
    physics transition, future-state input, or base prediction objective.
    """

    # A valid B2 default, unlike ACT's generic 100-step / 100-action default.
    chunk_size: int = 32
    n_action_steps: int = 1
    temporal_ensemble_coeff: float | None = None

    action_representation: str = "hermite_joint_trajectory"
    micro_horizon: int = 4
    num_micro_chunks: int = 8
    replan_every_step: bool = True
    use_latest_chunk_only: bool = True
    predict_joint_velocity: bool = True
    control_dt_s: float = 0.02
    max_joint_delta: float | None = None
    max_joint_velocity: float | None = None

    # Direct B2 supervision.  ``knot_loss`` supervises the terminal action of
    # each Hermite segment (plus the initial desired arm velocity); BC remains
    # the full 50-Hz action15 loss used for comparison with C0.
    action_bc_weight: float = 1.0
    knot_loss_weight: float = 1.0
    integration_weight: float = 0.0
    smoothness_weight: float = 0.0
    knot_decoder_hidden_dim: int = 256

    # Explicitly seal the B2 boundary.  B3/B4 must use distinct policy types,
    # not silently turn on an auxiliary or reaction mechanism here.
    base_auxiliary_enabled: bool = False
    expected_state_dim: int | None = None
    expected_action_dim: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_spline_hyperparameters()

    def _validate_spline_hyperparameters(self) -> None:
        if self.action_representation != "hermite_joint_trajectory":
            raise ValueError(
                "Spline ACT requires action_representation='hermite_joint_trajectory'"
            )
        if self.n_action_steps != 1:
            raise ValueError("Spline ACT requires n_action_steps=1 for latest-plan execution")
        if self.temporal_ensemble_coeff is not None:
            raise ValueError("Spline ACT is incompatible with ACT temporal ensembling")
        if not self.replan_every_step or not self.use_latest_chunk_only:
            raise ValueError("Spline ACT requires replan_every_step and use_latest_chunk_only")
        if self.micro_horizon <= 0 or self.num_micro_chunks <= 0:
            raise ValueError("micro_horizon and num_micro_chunks must be positive")
        if self.micro_horizon * self.num_micro_chunks != self.chunk_size:
            raise ValueError("Spline ACT chunk_size must equal micro_horizon * num_micro_chunks")
        if not self.predict_joint_velocity:
            raise ValueError(
                "Spline ACT requires predict_joint_velocity=True for canonical Left15 action15"
            )
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if self.max_joint_delta is not None and self.max_joint_delta <= 0.0:
            raise ValueError("max_joint_delta must be positive when set")
        if self.max_joint_velocity is not None and self.max_joint_velocity <= 0.0:
            raise ValueError("max_joint_velocity must be positive when set")
        if self.knot_decoder_hidden_dim <= 0:
            raise ValueError("knot_decoder_hidden_dim must be positive")
        loss_weights = {
            "action_bc_weight": self.action_bc_weight,
            "knot_loss_weight": self.knot_loss_weight,
            "integration_weight": self.integration_weight,
            "smoothness_weight": self.smoothness_weight,
        }
        if any(value < 0.0 for value in loss_weights.values()):
            raise ValueError(f"Spline ACT loss weights must be non-negative, got {loss_weights}")
        # ``SplineACTConfig`` is the sealed B2 type.  B3 has its own registered
        # subclass and intentionally reuses the Hermite checks above, so do not
        # make ``base_auxiliary_enabled`` a hidden B2 switch.  The exact-class
        # guard keeps a B2 config (including a hand-written config JSON) from
        # enabling an auxiliary head while allowing that separate type to
        # validate its own mandatory auxiliary contract.
        if self.base_auxiliary_enabled and self.__class__ is SplineACTConfig:
            raise ValueError(
                "B2 Spline ACT has no base auxiliary head; use the future B3 policy type instead"
            )

    def validate_features(self) -> None:
        super().validate_features()
        if self.robot_state_feature is None:
            raise ValueError("Spline ACT requires observation.state")
        if self.action_feature is None:
            raise ValueError("Spline ACT requires action")
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
        if self.robot_state_feature.shape != (8,):
            raise ValueError(
                "B2 Spline ACT requires canonical state8 (arm position7 + gripper position1); "
                f"got {self.robot_state_feature.shape}"
            )
        if self.action_feature.shape != (15,):
            raise ValueError(
                "B2 Spline ACT requires canonical action15: arm position7 + desired velocity7 + "
                f"gripper position1; got {self.action_feature.shape}"
            )
        if not self.image_features:
            raise ValueError("B2 Spline ACT keeps the visual ACT condition and requires at least one image")

    @property
    def action_feature_name(self) -> str:
        return ACTION

    @property
    def robot_state_feature_name(self) -> str:
        return OBS_STATE


__all__ = ["SplineACTConfig"]
