from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_spacepolicy_act import ReactionAwareACTConfig, ReactionAwareACTPolicy
from lerobot_policy_spacepolicy_act.reaction_model import (
    JOINT_POSITION_SLICE,
    JOINT_VELOCITY_SLICE,
    LEFT_ARM_INDICES,
)


def _features(action_dim: int = 15):
    return {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
        f"{OBS_IMAGES}.main": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
    }, {ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))}


def _small_act_kwargs():
    inputs, outputs = _features()
    return {
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
    }


def _diagnostics(batch_size: int) -> dict[str, torch.Tensor]:
    base_pose = torch.zeros(batch_size, 7)
    base_pose[:, 6] = 1.0
    return {
        "diagnostics.base_pose": base_pose,
        "diagnostics.base_velocity": torch.zeros(batch_size, 6),
        "diagnostics.raw_joint_state": torch.zeros(batch_size, 26),
        "diagnostics.measured_joint_velocity": torch.zeros(batch_size, 26),
    }


def test_disabled_reaction_plugin_matches_native_act_output() -> None:
    torch.manual_seed(4)
    native = ACTPolicy(ACTConfig(**_small_act_kwargs()))
    plugin = ReactionAwareACTPolicy(ReactionAwareACTConfig(**_small_act_kwargs(), reaction_aware=False))
    plugin.model.load_state_dict(native.model.state_dict())
    batch = {
        OBS_STATE: torch.randn(1, 8),
        f"{OBS_IMAGES}.main": torch.rand(1, 3, 32, 32),
    }

    native.eval()
    plugin.eval()
    with torch.no_grad():
        expected = native.predict_action_chunk(batch)
        actual = plugin.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 4, 15)


def test_reaction_aware_act_rolls_two_hermite_micro_chunks_and_backpropagates(tmp_path) -> None:
    torch.manual_seed(5)
    config = ReactionAwareACTConfig(
        **_small_act_kwargs(),
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        reaction_hidden_dim=16,
        reaction_num_layers=1,
        freeze_reaction_model=False,
    )
    policy = ReactionAwareACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(1, 8),
        f"{OBS_IMAGES}.main": torch.rand(1, 3, 32, 32),
        # Enabled Reaction-Aware ACT asks LeRobot for [-1, 0, 1, 2, 3]: the
        # first command is state history, the remaining four are ACT targets.
        ACTION: torch.zeros(1, 5, 15),
        "action_is_pad": torch.zeros(1, 5, dtype=torch.bool),
        **_diagnostics(1),
    }
    batch[ACTION][0, 0] = torch.arange(15, dtype=torch.float32)

    assert config.action_delta_indices == [-1, 0, 1, 2, 3]
    previous, target, target_pad = policy._split_reaction_training_action_window(batch)
    torch.testing.assert_close(previous, torch.arange(15, dtype=torch.float32).unsqueeze(0))
    assert target.shape == (1, 4, 15)
    assert target_pad.shape == (1, 4)

    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert set(metrics) >= {"l1_loss", "integration_loss", "smoothness_loss"}
    loss.backward()
    assert policy.micro_chunk_decoder is not None
    assert any(parameter.grad is not None for parameter in policy.micro_chunk_decoder.parameters())

    policy.eval()
    with torch.no_grad():
        chunk = policy.predict_action_chunk({key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}})
    assert chunk.shape == (1, 4, 15)
    assert len(policy._last_reaction_predictions) == 2
    arm_index = torch.tensor(LEFT_ARM_INDICES)
    first_end = chunk[:, 1]
    first_next_state = policy._last_reaction_predictions[0].next_robot_state
    torch.testing.assert_close(
        first_next_state[:, JOINT_POSITION_SLICE][:, arm_index], first_end[:, :7]
    )
    torch.testing.assert_close(
        first_next_state[:, JOINT_VELOCITY_SLICE][:, arm_index], first_end[:, 7:14]
    )
    action = policy.select_action({key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}})
    assert action.shape == (1, 15)

    # The bundled policy checkpoint carries both the ACT proposal path and the
    # reaction model; loading must not require the original external R0 path.
    policy.save_pretrained(tmp_path)
    restored = ReactionAwareACTPolicy.from_pretrained(tmp_path)
    assert restored.reaction_model is not None
    torch.testing.assert_close(
        restored.reaction_model.transition_head.weight,
        policy.reaction_model.transition_head.weight,
    )


def test_reaction_aware_frozen_mode_allows_a_bundled_policy_checkpoint() -> None:
    # The launcher requires an external reaction checkpoint for a fresh run, but
    # config construction must also allow a saved policy checkpoint whose reaction
    # weights are bundled in its own state_dict.
    config = ReactionAwareACTConfig(
        **_small_act_kwargs(),
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        freeze_reaction_model=True,
    )
    assert config.reaction_model_checkpoint is None


def test_reaction_aware_config_cannot_disable_the_precontact_admission_gate() -> None:
    with pytest.raises(ValueError, match="require_precontact_control_state"):
        ReactionAwareACTConfig(
            **_small_act_kwargs(),
            reaction_aware=True,
            action_representation="hermite_joint_trajectory",
            micro_horizon=2,
            num_micro_chunks=2,
            freeze_reaction_model=False,
            require_precontact_control_state=False,
        )
    with pytest.raises(ValueError, match="canonical precontact_control_states"):
        ReactionAwareACTConfig(
            **_small_act_kwargs(),
            reaction_aware=True,
            action_representation="hermite_joint_trajectory",
            micro_horizon=2,
            num_micro_chunks=2,
            freeze_reaction_model=False,
            precontact_control_states=(13, 14),
        )


def test_reaction_aware_mode_rejects_nonvisual_or_privileged_task_configuration() -> None:
    inputs, outputs = _features()
    inputs.pop(f"{OBS_IMAGES}.main")
    config = ReactionAwareACTConfig(
        device="cpu",
        input_features=inputs,
        output_features=outputs,
        chunk_size=4,
        n_action_steps=1,
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        freeze_reaction_model=False,
    )
    with pytest.raises(ValueError, match="at least one image"):
        config.validate_features()

    with pytest.raises(ValueError, match="audit-only"):
        ReactionAwareACTConfig(
            **_small_act_kwargs(),
            reaction_aware=True,
            action_representation="hermite_joint_trajectory",
            micro_horizon=2,
            num_micro_chunks=2,
            freeze_reaction_model=False,
            task_state_mode="relative_target_pose",
        )


def test_reaction_training_uses_zero_history_only_for_a_padded_episode_start() -> None:
    config = ReactionAwareACTConfig(
        **_small_act_kwargs(),
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        freeze_reaction_model=False,
    )
    policy = ReactionAwareACTPolicy(config)
    actions = torch.full((1, 5, 15), 7.0)
    padding = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)
    previous, target, target_pad = policy._split_reaction_training_action_window(
        {ACTION: actions, "action_is_pad": padding}
    )

    torch.testing.assert_close(previous, torch.zeros_like(previous))
    torch.testing.assert_close(target, actions[:, 1:])
    torch.testing.assert_close(target_pad, padding[:, 1:])


def test_reaction_aware_policy_remains_conditioned_on_visual_observation() -> None:
    """V1 reuses the current RGB latent for every causal micro-chunk."""

    torch.manual_seed(21)
    config = ReactionAwareACTConfig(
        **_small_act_kwargs(),
        reaction_aware=True,
        action_representation="hermite_joint_trajectory",
        micro_horizon=2,
        num_micro_chunks=2,
        reaction_hidden_dim=16,
        reaction_num_layers=1,
        freeze_reaction_model=False,
    )
    policy = ReactionAwareACTPolicy(config).eval()
    reaction_state = torch.zeros(1, 80)
    reaction_state[:, 6] = 1.0
    common = {
        OBS_STATE: torch.zeros(1, 8),
        "observation.reaction_state": reaction_state,
    }
    with torch.no_grad():
        dark = policy.predict_action_chunk(
            {**common, f"{OBS_IMAGES}.main": torch.zeros(1, 3, 32, 32)}
        )
        bright = policy.predict_action_chunk(
            {**common, f"{OBS_IMAGES}.main": torch.ones(1, 3, 32, 32)}
        )

    assert not torch.allclose(dark, bright)
