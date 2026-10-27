"""Configuration for B3: B2 Spline ACT plus a train-only base auxiliary head.

The B3 ablation is deliberately a *new* policy type.  It keeps B2's visual ACT
and Hermite action decoder, then adds a head which predicts a four-control-step
future base delta/twist from the emitted action micro-chunk.  The prediction is
never consumed by the action decoder, there is no reaction checkpoint, and no
base/target/EE diagnostic is part of the online policy input contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from .configuration_spline_act import SplineACTConfig


@PreTrainedConfig.register_subclass("base_aux_spline_act")
@dataclass
class BaseAuxiliarySplineACTConfig(SplineACTConfig):
    """Sealed B3 ACT configuration.

    ``base_auxiliary_horizon`` defines both the action prefix delivered to the
    auxiliary head and the future diagnostic label offset.  It must be a whole
    Hermite micro-chunk, which makes B3 directly comparable with B2/B4's
    ``K=4`` contract without introducing a rollout or an R0 dependency.
    """

    base_auxiliary_enabled: bool = True
    base_auxiliary_horizon: int = 4
    base_auxiliary_hidden_dim: int = 256
    base_auxiliary_loss_weight: float = 1.0
    base_auxiliary_translation_weight: float = 1.0
    base_auxiliary_rotation_weight: float = 1.0
    base_auxiliary_twist_weight: float = 1.0
    base_auxiliary_label_schema: str = "body_delta_se3_future_body_twist_v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_base_auxiliary_hyperparameters()

    def _validate_base_auxiliary_hyperparameters(self) -> None:
        if not self.base_auxiliary_enabled:
            raise ValueError(
                "base_aux_spline_act requires base_auxiliary_enabled=True; "
                "use spline_act for the B2 no-auxiliary ablation"
            )
        if self.base_auxiliary_horizon <= 0:
            raise ValueError("base_auxiliary_horizon must be positive")
        if self.base_auxiliary_horizon != self.micro_horizon:
            raise ValueError(
                "B3 base_auxiliary_horizon must equal micro_horizon so the head "
                "uses exactly one emitted Hermite micro-chunk"
            )
        if self.base_auxiliary_horizon > self.chunk_size:
            raise ValueError("base_auxiliary_horizon must not exceed chunk_size")
        if self.base_auxiliary_hidden_dim <= 0:
            raise ValueError("base_auxiliary_hidden_dim must be positive")
        weights = {
            "base_auxiliary_loss_weight": self.base_auxiliary_loss_weight,
            "base_auxiliary_translation_weight": self.base_auxiliary_translation_weight,
            "base_auxiliary_rotation_weight": self.base_auxiliary_rotation_weight,
            "base_auxiliary_twist_weight": self.base_auxiliary_twist_weight,
        }
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError(f"base auxiliary loss weights must be non-negative, got {weights}")
        if self.base_auxiliary_label_schema != "body_delta_se3_future_body_twist_v1":
            raise ValueError(
                "B3 only supports base_auxiliary_label_schema="
                "'body_delta_se3_future_body_twist_v1'"
            )


__all__ = ["BaseAuxiliarySplineACTConfig"]
