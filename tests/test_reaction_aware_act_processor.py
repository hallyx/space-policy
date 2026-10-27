"""Tests for raw-state retention at the ACT processor boundary."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
import lerobot.policies.factory as policy_factory
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_spacepolicy_dit import ReactionAwareACTConfig, ReactionAwareACTPolicy
from lerobot_policy_spacepolicy_dit.processor_reaction_aware_act import (
    make_reaction_aware_act_pre_post_processors,
)
from lerobot_policy_spacepolicy_dit.reaction_phase_gate import ReactionAwarePhaseGateError


def _config() -> ReactionAwareACTConfig:
    return ReactionAwareACTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
            f"{OBS_IMAGES}.main": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (15,))},
        chunk_size=4,
        n_action_steps=1,
        use_vae=False,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        reaction_hidden_dim=16,
        reaction_num_layers=1,
        freeze_reaction_model=False,
    )


def _training_batch() -> dict[str, torch.Tensor]:
    base_pose = torch.zeros(1, 7)
    base_pose[:, 6] = 1.0
    return {
        OBS_STATE: torch.zeros(1, 8),
        f"{OBS_IMAGES}.main": torch.zeros(1, 3, 32, 32),
        ACTION: torch.zeros(1, 5, 15),
        "action_is_pad": torch.zeros(1, 5, dtype=torch.bool),
        "diagnostics.base_pose": base_pose,
        "diagnostics.base_velocity": torch.full((1, 6), 2.0),
        "diagnostics.raw_joint_state": torch.full((1, 26), 3.0),
        "diagnostics.measured_joint_velocity": torch.full((1, 26), 4.0),
        # Admission-only metadata. It must be checked before transition
        # conversion and must not reach the policy/R0 batch.
        "diagnostics.control_state": torch.tensor([13], dtype=torch.int64),
        # These simulator-derived privileged fields must stay dropped at the
        # generic processor boundary, not merely ignored downstream.
        "diagnostics.relative_target_pose": torch.full((1, 7), 5.0),
        "diagnostics.delta_ee_target": torch.full((1, 7), 6.0),
        "diagnostics.ee_pose": torch.full((1, 14), 7.0),
    }


def test_processor_retains_only_allowed_raw_reaction_diagnostics_for_policy_training() -> None:
    config = _config()
    # This is the *upstream* factory entrypoint used by lerobot_train.  The
    # plugin shim must intercept the ACT subclass before its generic ACT branch.
    preprocessor, _ = policy_factory.make_pre_post_processors(config)
    batch = _training_batch()

    processed = preprocessor(batch)

    for key in (
        "diagnostics.base_pose",
        "diagnostics.base_velocity",
        "diagnostics.raw_joint_state",
        "diagnostics.measured_joint_velocity",
    ):
        torch.testing.assert_close(processed[key], batch[key])
    for key in (
        "diagnostics.control_state",
        "diagnostics.relative_target_pose",
        "diagnostics.delta_ee_target",
        "diagnostics.ee_pose",
    ):
        assert key not in processed
    assert not any("control_state" in key for key in processed)

    # This crosses the real processor boundary before the enabled policy, so a
    # future change cannot silently reintroduce the missing-diagnostics failure.
    policy = ReactionAwareACTPolicy(config)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)


def test_loaded_reaction_processor_reattaches_raw_diagnostic_converter(tmp_path) -> None:
    config = _config()
    preprocessor, postprocessor = make_reaction_aware_act_pre_post_processors(config)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    # This exercises the same ``pretrained_path`` branch used by resume.  The
    # converter itself is not serialized with pipeline steps, so it must be
    # supplied again by the reaction-aware factory.
    resumed_preprocessor, _ = policy_factory.make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
    )
    processed = resumed_preprocessor(_training_batch())
    assert "diagnostics.base_pose" in processed
    assert "diagnostics.raw_joint_state" in processed
    assert "diagnostics.control_state" not in processed
    assert "diagnostics.relative_target_pose" not in processed


@pytest.mark.parametrize(
    ("control_state", "message"),
    [
        (None, "required"),
        (torch.tensor([12], dtype=torch.int64), "no post-contact fallback"),
        (torch.tensor([13.5], dtype=torch.float32), "finite integer"),
    ],
)
def test_reaction_processor_fail_closes_before_policy_on_invalid_phase(
    control_state: torch.Tensor | None,
    message: str,
) -> None:
    config = _config()
    preprocessor, _ = policy_factory.make_pre_post_processors(config)
    batch = _training_batch()
    if control_state is None:
        del batch["diagnostics.control_state"]
    else:
        batch["diagnostics.control_state"] = control_state

    with pytest.raises(ReactionAwarePhaseGateError, match=message):
        preprocessor(batch)
