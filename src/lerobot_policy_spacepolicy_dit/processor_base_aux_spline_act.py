"""Processor and loader shims for the B3 train-label-only diagnostics.

LeRobot's normal ACT converter deliberately drops arbitrary ``diagnostics.*``
columns.  B3 therefore preserves exactly two requested diagnostic windows and
their pad masks.  They survive only long enough for ``forward`` to form an
auxiliary loss; the model-input whitelist in ``modeling_base_aux_spline_act``
omits them, and inference has no dependency on them.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import torch

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.converters import batch_to_transition
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .base_auxiliary import BASE_AUXILIARY_DIAGNOSTIC_KEYS
from .configuration_base_aux_spline_act import BaseAuxiliarySplineACTConfig


_BASE_AUXILIARY_LABEL_BATCH_KEYS = (
    *BASE_AUXILIARY_DIAGNOSTIC_KEYS,
    "diagnostics.base_pose_is_pad",
    "diagnostics.base_velocity_is_pad",
)


def base_auxiliary_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Keep the sealed B3 label allow-list through the ACT processor.

    In particular this does *not* preserve target pose, EE pose, q/dq, control
    state, reward, or arbitrary diagnostic columns.  The two windows are not
    declared policy features, so ACT's normalizer leaves their raw physical
    units intact and the B3 backbone never receives them.
    """

    transition = batch_to_transition(batch)
    observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
    for key in _BASE_AUXILIARY_LABEL_BATCH_KEYS:
        if key in batch:
            observation[key] = batch[key]
    transition[TransitionKey.OBSERVATION] = observation
    return transition


def make_base_aux_spline_act_pre_post_processors(
    config: BaseAuxiliarySplineACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    pretrained_path: str | None = None,
    **kwargs: Any,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build/load B3 processors while reattaching the label converter."""

    if pretrained_path is not None:
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=base_auxiliary_batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        return preprocessor, postprocessor

    preprocessor, postprocessor = make_act_pre_post_processors(config=config, dataset_stats=dataset_stats)
    preprocessor.to_transition = base_auxiliary_batch_to_transition
    return preprocessor, postprocessor


def _base_auxiliary_delta_timestamps(
    current: dict[str, list] | None,
    *,
    config: BaseAuxiliarySplineACTConfig,
    dataset_meta: Any,
) -> dict[str, list]:
    missing = [key for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS if key not in dataset_meta.features]
    if missing:
        raise ValueError(
            "B3 canonical action15/state8 training requires train-label diagnostics "
            f"{list(BASE_AUXILIARY_DIAGNOSTIC_KEYS)}, dataset is missing {missing}"
        )
    result = dict(current or {})
    offsets = [0.0, float(config.base_auxiliary_horizon) / float(dataset_meta.fps)]
    for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS:
        existing = result.get(key)
        if existing is not None and list(existing) != offsets:
            raise ValueError(
                f"B3 diagnostic window conflict for {key}: expected {offsets}, got {existing}"
            )
        result[key] = offsets
    return result


def install_base_aux_spline_act_runtime_shims() -> None:
    """Install narrow B3 hooks for upstream processor and dataset factories.

    Both hooks are type-gated.  Native ACT, B2 Spline ACT, and Reaction-Aware
    ACT continue into the previously installed factory/resolver unchanged.
    """

    import lerobot.datasets.factory as dataset_factory_module
    import lerobot.policies as policies_module
    import lerobot.policies.factory as policy_factory_module

    current_resolver = dataset_factory_module.resolve_delta_timestamps
    if not getattr(current_resolver, "_spacepolicy_base_aux_spline_act_resolver", False):

        @wraps(current_resolver)
        def base_auxiliary_resolver(config: Any, dataset_meta: Any):
            resolved = current_resolver(config, dataset_meta)
            if isinstance(config, BaseAuxiliarySplineACTConfig):
                return _base_auxiliary_delta_timestamps(
                    resolved,
                    config=config,
                    dataset_meta=dataset_meta,
                )
            return resolved

        setattr(base_auxiliary_resolver, "_spacepolicy_base_aux_spline_act_resolver", True)
        dataset_factory_module.resolve_delta_timestamps = base_auxiliary_resolver

    current_factory = policy_factory_module.make_pre_post_processors
    if getattr(current_factory, "_spacepolicy_base_aux_spline_act_factory", False):
        return

    @wraps(current_factory)
    def base_auxiliary_factory(
        policy_cfg: Any,
        pretrained_path: str | None = None,
        **factory_kwargs: Any,
    ) -> tuple[Any, Any]:
        if isinstance(policy_cfg, BaseAuxiliarySplineACTConfig):
            return make_base_aux_spline_act_pre_post_processors(
                config=policy_cfg,
                pretrained_path=pretrained_path,
                dataset_stats=factory_kwargs.pop("dataset_stats", None),
                **factory_kwargs,
            )
        return current_factory(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            **factory_kwargs,
        )

    setattr(base_auxiliary_factory, "_spacepolicy_base_aux_spline_act_factory", True)
    policy_factory_module.make_pre_post_processors = base_auxiliary_factory
    policies_module.make_pre_post_processors = base_auxiliary_factory
    # Same import-time detail as Reaction-Aware ACT: lerobot_train holds a
    # module-level factory reference before plugin discovery happens.
    for module_name in ("lerobot.scripts.lerobot_train", "__main__"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "make_pre_post_processors", None) is current_factory:
            setattr(module, "make_pre_post_processors", base_auxiliary_factory)


__all__ = [
    "BASE_AUXILIARY_DIAGNOSTIC_KEYS",
    "base_auxiliary_batch_to_transition",
    "install_base_aux_spline_act_runtime_shims",
    "make_base_aux_spline_act_pre_post_processors",
]
