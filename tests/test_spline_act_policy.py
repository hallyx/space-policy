"""Contract tests for the B2 no-reaction Spline ACT policy type."""

from __future__ import annotations

import types

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import get_policy_class
import lerobot.policies.factory as policy_factory
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_spacepolicy_dit import SplineACTConfig, SplineACTPolicy


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
            f"{OBS_IMAGES}.main": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (15,))},
    )


def _small_kwargs() -> dict[str, object]:
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
        "micro_horizon": 2,
        "num_micro_chunks": 2,
        "knot_decoder_hidden_dim": 16,
    }


def _config(**overrides: object) -> SplineACTConfig:
    values = _small_kwargs()
    values.update(overrides)
    return SplineACTConfig(**values)


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    state = torch.zeros(batch_size, 8)
    state[:, :7] = torch.linspace(-0.3, 0.3, 7)
    state[:, 7] = 0.2
    return {
        OBS_STATE: state,
        f"{OBS_IMAGES}.main": torch.rand(batch_size, 3, 32, 32),
        ACTION: torch.randn(batch_size, 4, 15),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_spline_act_is_an_independent_registered_policy_type() -> None:
    config = _config()

    assert PreTrainedConfig.get_choice_class("spline_act") is SplineACTConfig
    assert get_policy_class("spline_act") is SplineACTPolicy
    assert config.type == "spline_act"
    assert config.action_delta_indices == [0, 1, 2, 3]
    assert not hasattr(config, "reaction_model_checkpoint")


def test_spline_act_seals_b2_latest_plan_without_base_auxiliary_or_reaction() -> None:
    with pytest.raises(ValueError, match="n_action_steps=1"):
        _config(n_action_steps=2)
    with pytest.raises(ValueError, match="temporal ensembling"):
        _config(temporal_ensemble_coeff=0.01)
    with pytest.raises(ValueError, match="base auxiliary"):
        _config(base_auxiliary_enabled=True)
    with pytest.raises(ValueError, match="action_representation"):
        _config(action_representation="original")


def test_spline_act_decodes_hermite_knots_and_backpropagates_without_reaction_model() -> None:
    torch.manual_seed(31)
    policy = SplineACTPolicy(_config())
    batch = _batch()

    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert set(metrics) >= {"l1_loss", "knot_loss", "endpoint_loss", "initial_velocity_loss"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.knot_decoder.parameters())
    assert not hasattr(policy, "reaction_model")
    assert not hasattr(policy, "_last_reaction_predictions")
    assert not any("reaction" in name for name, _ in policy.named_modules())

    policy.eval()
    observation = {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}
    with torch.no_grad():
        chunk = policy.predict_action_chunk(observation)
    assert chunk.shape == (1, 4, 15)
    assert policy._last_spline_endpoints_raw is not None
    assert policy._last_spline_start_positions_raw is not None
    assert policy._last_spline_start_velocities_raw is not None

    endpoints = policy._last_spline_endpoints_raw
    starts = policy._last_spline_start_positions_raw
    start_velocities = policy._last_spline_start_velocities_raw
    torch.testing.assert_close(starts[:, 0], batch[OBS_STATE][:, :7])
    torch.testing.assert_close(starts[:, 1], endpoints[:, 0, :7])
    torch.testing.assert_close(start_velocities[:, 1], endpoints[:, 0, 7:14])
    # The final generated sample of every micro segment is exactly its decoded
    # Hermite endpoint, including desired velocity and the linearly continuous
    # gripper position.
    torch.testing.assert_close(chunk[:, 1], endpoints[:, 0])
    torch.testing.assert_close(chunk[:, 3], endpoints[:, 1])


def test_spline_act_ignores_privileged_task_and_future_state_fields() -> None:
    torch.manual_seed(32)
    policy = SplineACTPolicy(_config()).eval()
    batch = _batch()
    observation = {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}
    privileged = {
        **observation,
        "diagnostics.relative_target_pose": torch.full((1, 7), 1.0e6),
        "diagnostics.ee_pose": torch.full((1, 14), -1.0e6),
        "diagnostics.base_pose": torch.full((1, 7), 5.0e5),
        "future_robot_state": torch.full((1, 80), -7.0e5),
    }
    with torch.no_grad():
        expected = policy.predict_action_chunk(observation)
        actual = policy.predict_action_chunk(privileged)
    torch.testing.assert_close(actual, expected)


def test_spline_checkpoint_is_self_contained_and_uses_standard_act_processor(tmp_path) -> None:
    torch.manual_seed(33)
    config = _config()
    policy = SplineACTPolicy(config)
    policy.save_pretrained(tmp_path)

    restored = SplineACTPolicy.from_pretrained(tmp_path)
    assert restored.config.type == "spline_act"
    assert not hasattr(restored, "reaction_model")
    torch.testing.assert_close(
        restored.knot_decoder.endpoint_network[-1].weight,
        policy.knot_decoder.endpoint_network[-1].weight,
    )

    # No B2-specific raw diagnostic converter is needed: the unmodified ACT
    # processor is the intentional ABI boundary for this ablation.
    preprocessor, postprocessor = policy_factory.make_pre_post_processors(config)
    processed = preprocessor(_batch())
    assert processed[ACTION].shape == (1, 4, 15)
    assert processed["action_is_pad"].shape == (1, 4)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    resumed_preprocessor, _ = policy_factory.make_pre_post_processors(
        config, pretrained_path=str(tmp_path)
    )
    resumed = resumed_preprocessor(_batch())
    assert resumed[ACTION].shape == (1, 4, 15)


def test_spline_latest_plan_requeries_act_backbone_on_every_action() -> None:
    policy = SplineACTPolicy(_config()).eval()
    calls = 0

    def fake_predict(_self: SplineACTPolicy, _batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.zeros(1, 4, 15)

    policy.predict_action_chunk = types.MethodType(fake_predict, policy)  # type: ignore[method-assign]
    observation = _batch()
    observation.pop(ACTION)
    observation.pop("action_is_pad")
    first = policy.select_action(observation)
    second = policy.select_action(observation)

    assert calls == 2
    assert first.shape == second.shape == (1, 15)


def test_importing_spline_type_does_not_change_native_act_output() -> None:
    torch.manual_seed(34)
    kwargs = {
        key: value
        for key, value in _small_kwargs().items()
        if key not in {"micro_horizon", "num_micro_chunks", "knot_decoder_hidden_dim"}
    }
    native = ACTPolicy(ACTConfig(**kwargs))
    # The test imports the plugin at module import time.  Native ACT must still
    # have its original implementation and state-dict ABI.
    reference = ACTPolicy(ACTConfig(**kwargs))
    reference.model.load_state_dict(native.model.state_dict())
    batch = _batch()
    observation = {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}
    native.eval()
    reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            native.predict_action_chunk(observation), reference.predict_action_chunk(observation)
        )
