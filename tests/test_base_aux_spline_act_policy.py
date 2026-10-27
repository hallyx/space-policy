"""Minimal B3 contracts: label-only base diagnostics, no R0, no rollout."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
import lerobot.policies.factory as policy_factory
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_spacepolicy_act import (
    BaseAuxiliarySplineACTConfig,
    BaseAuxiliarySplineACTPolicy,
    SplineACTConfig,
)
from lerobot_policy_spacepolicy_act.base_auxiliary import (
    ActionConditionedBaseAuxiliaryHead,
    base_auxiliary_loss,
    derive_base_auxiliary_labels,
)


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
            f"{OBS_IMAGES}.main": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (15,))},
    )


def _config(**overrides: object) -> BaseAuxiliarySplineACTConfig:
    inputs, outputs = _features()
    values: dict[str, object] = {
        "device": "cpu",
        "input_features": inputs,
        "output_features": outputs,
        "chunk_size": 4,
        "n_action_steps": 1,
        "use_vae": False,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": None,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "micro_horizon": 2,
        "num_micro_chunks": 2,
        "knot_decoder_hidden_dim": 16,
        "base_auxiliary_horizon": 2,
        "base_auxiliary_hidden_dim": 16,
    }
    values.update(overrides)
    return BaseAuxiliarySplineACTConfig(**values)


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    pose = torch.zeros(batch_size, 2, 7)
    pose[..., 6] = 1.0
    pose[:, 1, 0] = 0.02
    velocity = torch.zeros(batch_size, 2, 6)
    velocity[:, 1, 0] = 0.10
    return {
        OBS_STATE: torch.cat(
            (torch.linspace(-0.3, 0.3, 7).repeat(batch_size, 1), torch.full((batch_size, 1), 0.2)),
            dim=-1,
        ),
        f"{OBS_IMAGES}.main": torch.rand(batch_size, 3, 32, 32),
        ACTION: torch.randn(batch_size, 4, 15),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        "diagnostics.base_pose": pose,
        "diagnostics.base_velocity": velocity,
        "diagnostics.base_pose_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
        "diagnostics.base_velocity_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
    }


def test_b3_is_registered_and_b2_cannot_enable_the_auxiliary() -> None:
    config = _config()
    assert PreTrainedConfig.get_choice_class("base_aux_spline_act") is BaseAuxiliarySplineACTConfig
    assert get_policy_class("base_aux_spline_act") is BaseAuxiliarySplineACTPolicy
    assert config.type == "base_aux_spline_act"
    assert config.action_delta_indices == [0, 1, 2, 3]
    assert not hasattr(config, "reaction_model_checkpoint")

    with pytest.raises(ValueError, match="requires base_auxiliary_enabled=True"):
        _config(base_auxiliary_enabled=False)
    with pytest.raises(ValueError, match="must equal micro_horizon"):
        _config(base_auxiliary_horizon=1)
    b3_values = dict(_config().__dict__)
    for key in (
        "base_auxiliary_horizon",
        "base_auxiliary_hidden_dim",
        "base_auxiliary_loss_weight",
        "base_auxiliary_translation_weight",
        "base_auxiliary_rotation_weight",
        "base_auxiliary_twist_weight",
        "base_auxiliary_label_schema",
    ):
        b3_values.pop(key)
    with pytest.raises(ValueError, match="B2 Spline ACT has no base auxiliary"):
        SplineACTConfig(**{**b3_values, "base_auxiliary_enabled": True})


def test_b3_derives_body_label_and_masks_edge_clamped_diagnostics() -> None:
    batch = _batch()
    labels = derive_base_auxiliary_labels(
        batch["diagnostics.base_pose"], batch["diagnostics.base_velocity"]
    )
    torch.testing.assert_close(labels.target_body_delta_se3[:, :3], torch.tensor([[0.02, 0.0, 0.0]]))
    torch.testing.assert_close(labels.target_future_twist_body, torch.tensor([[0.10, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert bool(labels.valid.all())

    # Invalid edge labels are ignored rather than scoring clamped or even
    # malformed future diagnostics as a real transition.
    padded_pose = torch.zeros_like(batch["diagnostics.base_pose"])
    padded_velocity = torch.zeros_like(batch["diagnostics.base_velocity"])
    padded = derive_base_auxiliary_labels(
        padded_pose,
        padded_velocity,
        base_pose_is_pad=torch.ones(1, 2, dtype=torch.bool),
        base_velocity_is_pad=torch.ones(1, 2, dtype=torch.bool),
    )
    head = ActionConditionedBaseAuxiliaryHead(action_dim=15, horizon=2, hidden_dim=8)
    loss = base_auxiliary_loss(head(torch.zeros(1, 2, 15)), padded)
    assert loss.valid_samples.item() == 0
    assert loss.total.item() == 0.0

    # The auxiliary is genuinely action-conditioned, not a state-only base
    # regressor hidden behind the B3 name.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.network[0].weight[0, 0] = 1.0
        head.network[2].weight[0, 0] = 1.0
        head.network[4].weight[0, 0] = 1.0
    assert not torch.equal(
        head(torch.zeros(1, 2, 15)).body_delta_se3,
        head(torch.ones(1, 2, 15)).body_delta_se3,
    )


def test_b3_forward_backpropagates_but_prediction_never_feeds_action_decoder() -> None:
    torch.manual_seed(71)
    policy = BaseAuxiliarySplineACTPolicy(_config()).eval()
    batch = _batch()
    observation = {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}

    with torch.no_grad():
        before = policy.predict_action_chunk(observation)
    assert policy._last_base_auxiliary_prediction is None
    # Deliberately alter every auxiliary parameter.  An online action can only
    # change if the decoder consumes its prediction, which B3 forbids.
    with torch.no_grad():
        for parameter in policy.base_auxiliary_head.parameters():
            parameter.add_(0.5)
        after = policy.predict_action_chunk(observation)
    torch.testing.assert_close(after, before)

    policy.train()
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert set(metrics) >= {
        "base_auxiliary_loss",
        "base_auxiliary_translation_loss",
        "base_auxiliary_rotation_loss",
        "base_auxiliary_twist_loss",
        "base_auxiliary_valid_samples",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.base_auxiliary_head.parameters())
    assert policy._last_base_auxiliary_prediction is not None
    assert not hasattr(policy, "reaction_model")
    assert not any("reaction" in name for name, _ in policy.named_modules())


def test_b3_backbone_whitelist_and_processor_keep_only_label_diagnostics() -> None:
    policy = BaseAuxiliarySplineACTPolicy(_config()).eval()
    batch = _batch()
    privileged = {
        **batch,
        "diagnostics.relative_target_pose": torch.full((1, 2, 7), 1.0e6),
        "diagnostics.ee_pose": torch.full((1, 2, 14), -1.0e6),
        "future_robot_state": torch.full((1, 80), 7.0e5),
    }
    model_batch = policy._prepare_model_batch(
        privileged,
        action_targets=privileged[ACTION],
        action_is_pad=privileged["action_is_pad"],
    )
    assert set(model_batch) == {OBS_STATE, OBS_IMAGES, ACTION, "action_is_pad"}

    preprocessor, _ = policy_factory.make_pre_post_processors(policy.config)
    processed = preprocessor(privileged)
    assert "diagnostics.base_pose" in processed
    assert "diagnostics.base_velocity" in processed
    assert "diagnostics.base_pose_is_pad" in processed
    assert "diagnostics.base_velocity_is_pad" in processed
    assert "diagnostics.relative_target_pose" not in processed
    assert "diagnostics.ee_pose" not in processed
    assert "future_robot_state" not in processed


def test_b3_checkpoint_and_resumed_processor_keep_the_auxiliary_and_label_converter(tmp_path) -> None:
    torch.manual_seed(72)
    policy = BaseAuxiliarySplineACTPolicy(_config())
    policy.save_pretrained(tmp_path)
    restored = BaseAuxiliarySplineACTPolicy.from_pretrained(tmp_path)
    assert restored.config.type == "base_aux_spline_act"
    torch.testing.assert_close(
        restored.base_auxiliary_head.network[-1].weight,
        policy.base_auxiliary_head.network[-1].weight,
    )

    preprocessor, postprocessor = policy_factory.make_pre_post_processors(policy.config)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    resumed_preprocessor, _ = policy_factory.make_pre_post_processors(
        policy.config, pretrained_path=str(tmp_path)
    )
    resumed = resumed_preprocessor(_batch())
    assert resumed["diagnostics.base_pose"].shape == (1, 2, 7)
    assert resumed["diagnostics.base_velocity"].shape == (1, 2, 6)
