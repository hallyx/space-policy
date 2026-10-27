"""B4 = B3 train-only auxiliary + frozen R0, with no feedback leakage."""

from __future__ import annotations

import pytest
import torch

import lerobot.datasets.factory as dataset_factory
from lerobot.configs.types import FeatureType, PolicyFeature
import lerobot.policies.factory as policy_factory
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_spacepolicy_act import ReactionAwareACTConfig, ReactionAwareACTPolicy


def _config(**overrides: object) -> ReactionAwareACTConfig:
    inputs = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
        f"{OBS_IMAGES}.main": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
    }
    outputs = {ACTION: PolicyFeature(FeatureType.ACTION, (15,))}
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
        "reaction_aware": True,
        "action_representation": "hermite_joint_trajectory",
        "micro_horizon": 2,
        "num_micro_chunks": 2,
        "reaction_hidden_dim": 16,
        "reaction_num_layers": 1,
        "freeze_reaction_model": True,
        "base_auxiliary_enabled": True,
        "base_auxiliary_horizon": 2,
        "base_auxiliary_hidden_dim": 16,
    }
    values.update(overrides)
    return ReactionAwareACTConfig(**values)


def _batch() -> dict[str, torch.Tensor]:
    pose = torch.zeros(1, 2, 7)
    pose[..., 6] = 1.0
    pose[:, 1, 0] = 0.02
    twist = torch.zeros(1, 2, 6)
    twist[:, 1, 0] = 0.1
    return {
        OBS_STATE: torch.zeros(1, 8),
        f"{OBS_IMAGES}.main": torch.rand(1, 3, 32, 32),
        # [-1, 0, 1, 2, 3]: history + ordinary ACT target block.
        ACTION: torch.zeros(1, 5, 15),
        "action_is_pad": torch.zeros(1, 5, dtype=torch.bool),
        "diagnostics.base_pose": pose,
        "diagnostics.base_velocity": twist,
        "diagnostics.base_pose_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        "diagnostics.base_velocity_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        "diagnostics.raw_joint_state": torch.zeros(1, 26),
        "diagnostics.measured_joint_velocity": torch.zeros(1, 26),
        # Admission metadata and privileged fields must be discarded before the
        # policy forward, even though this direct fixture carries them.
        "diagnostics.control_state": torch.tensor([13], dtype=torch.int64),
        "diagnostics.relative_target_pose": torch.full((1, 7), 3.0),
        "diagnostics.ee_pose": torch.full((1, 14), -4.0),
    }


def _online_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}


def test_sealed_b4_auxiliary_is_label_only_and_r0_stays_frozen(tmp_path) -> None:
    torch.manual_seed(91)
    config = _config()
    policy = ReactionAwareACTPolicy(config).eval()
    assert policy.base_auxiliary_head is not None
    assert policy.reaction_model is not None
    assert all(not parameter.requires_grad for parameter in policy.reaction_model.parameters())
    batch = _batch()
    online = _online_batch(batch)

    # Neither the auxiliary head nor future diagnostic labels are used by
    # predict_action_chunk/R0.  The latter receives only base(t), checked by a
    # deliberately large t+K counterfactual.
    with torch.no_grad():
        expected = policy.predict_action_chunk(online)
        changed_future = dict(online)
        changed_future["diagnostics.base_pose"] = online["diagnostics.base_pose"].clone()
        changed_future["diagnostics.base_velocity"] = online["diagnostics.base_velocity"].clone()
        changed_future["diagnostics.base_pose"][:, 1, :3] = 1.0e5
        changed_future["diagnostics.base_velocity"][:, 1] = -1.0e5
        future_changed = policy.predict_action_chunk(changed_future)
        for parameter in policy.base_auxiliary_head.parameters():
            parameter.add_(0.5)
        head_changed = policy.predict_action_chunk(online)
    torch.testing.assert_close(future_changed, expected)
    torch.testing.assert_close(head_changed, expected)
    assert policy._last_base_auxiliary_prediction is None

    # ACT itself receives no raw diagnostics/targets/control label.  The raw
    # reaction state has its separate causal input path; auxiliary labels are
    # consumed only after the action chunk is generated.
    model_batch = policy._prepare_model_batch(
        batch,
        action_targets=batch[ACTION][:, 1:],
        action_is_pad=batch["action_is_pad"][:, 1:],
    )
    assert set(model_batch) == {OBS_STATE, OBS_IMAGES, ACTION, "action_is_pad"}

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
    assert all(parameter.grad is None for parameter in policy.reaction_model.parameters())
    assert policy._last_base_auxiliary_prediction is not None

    policy.save_pretrained(tmp_path)
    restored = ReactionAwareACTPolicy.from_pretrained(tmp_path)
    assert restored.config.base_auxiliary_enabled is True
    assert restored.base_auxiliary_head is not None
    torch.testing.assert_close(
        restored.base_auxiliary_head.network[-1].weight,
        policy.base_auxiliary_head.network[-1].weight,
    )


def test_b4_processor_and_loader_keep_only_current_r0_state_plus_base_label_window() -> None:
    config = _config()
    preprocessor, _ = policy_factory.make_pre_post_processors(config)
    processed = preprocessor(_batch())
    for key, shape in (
        ("diagnostics.base_pose", (1, 2, 7)),
        ("diagnostics.base_velocity", (1, 2, 6)),
        ("diagnostics.base_pose_is_pad", (1, 2)),
        ("diagnostics.base_velocity_is_pad", (1, 2)),
        ("diagnostics.raw_joint_state", (1, 26)),
        ("diagnostics.measured_joint_velocity", (1, 26)),
    ):
        assert processed[key].shape == shape
    for key in (
        "diagnostics.control_state",
        "diagnostics.relative_target_pose",
        "diagnostics.ee_pose",
    ):
        assert key not in processed

    class _Meta:
        fps = 50
        features = {
            ACTION: {},
            "diagnostics.base_pose": {},
            "diagnostics.base_velocity": {},
        }

    timestamps = dataset_factory.resolve_delta_timestamps(config, _Meta())
    assert timestamps[ACTION] == [-0.02, 0.0, 0.02, 0.04, 0.06]
    assert timestamps["diagnostics.base_pose"] == [0.0, 0.04]
    assert timestamps["diagnostics.base_velocity"] == [0.0, 0.04]


def test_b4_auxiliary_requires_frozen_r0_but_core_no_aux_test_path_remains_explicit() -> None:
    with pytest.raises(ValueError, match="freeze_reaction_model=True"):
        _config(freeze_reaction_model=False)

    core_config = _config(base_auxiliary_enabled=False, freeze_reaction_model=False)
    core_policy = ReactionAwareACTPolicy(core_config)
    assert core_policy.base_auxiliary_head is None

