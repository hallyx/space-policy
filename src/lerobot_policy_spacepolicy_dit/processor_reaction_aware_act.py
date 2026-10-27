"""Processor factory for the out-of-tree Reaction-Aware ACT policy."""

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
from .configuration_reaction_aware_act import ReactionAwareACTConfig
from .reaction_phase_gate import (
    REACTION_CONTROL_STATE_KEY,
    require_precontact_control_state,
)


# These are the four and only four canonical diagnostic fields needed to build
# the v1 raw 80D reaction state offline.  The generic LeRobot batch converter
# intentionally drops arbitrary top-level keys; preserving this allow-list here
# prevents it from silently losing the physical state while still excluding all
# target/EE diagnostics from the policy processor.
_RAW_REACTION_DIAGNOSTIC_KEYS = (
    "diagnostics.base_pose",
    "diagnostics.base_velocity",
    "diagnostics.raw_joint_state",
    "diagnostics.measured_joint_velocity",
)
_BASE_AUXILIARY_LABEL_PAD_KEYS = (
    "diagnostics.base_pose_is_pad",
    "diagnostics.base_velocity_is_pad",
)


def reaction_aware_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Admit pre-contact batches, then retain only sealed B4 fields.

    ``batch_to_transition`` recognizes ``observation.*`` fields but treats
    canonical ``diagnostics.*`` as unknown and discards them.  Place the strict
    allow-list into the transition observation dictionary so device transfer and
    the subsequent batch conversion retain it unchanged.  The ACT normalizer
    only normalizes configured features, so these raw fields stay in the units
    expected by the separately checkpointed Reaction model.

    ``diagnostics.control_state`` is intentionally checked *before* conversion
    and never copied into the transition.  It is an admission label rather than
    a policy/R0 input, so no downstream tensor can learn a phase shortcut or
    silently run past the validated pre-contact support.
    """

    require_precontact_control_state(
        batch.get(REACTION_CONTROL_STATE_KEY),
        source=f"offline batch {REACTION_CONTROL_STATE_KEY}",
    )
    transition = batch_to_transition(batch)
    observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
    for key in (*_RAW_REACTION_DIAGNOSTIC_KEYS, *_BASE_AUXILIARY_LABEL_PAD_KEYS):
        if key in batch:
            observation[key] = batch[key]
    transition[TransitionKey.OBSERVATION] = observation
    return transition


def _reaction_aware_base_auxiliary_delta_timestamps(
    current: dict[str, list] | None,
    *,
    config: ReactionAwareACTConfig,
    dataset_meta: Any,
) -> dict[str, list]:
    """Add exactly ``[t, t + K]`` base label windows for sealed B4 training.

    The normal LeRobot resolver only assigns temporal windows to ``action`` and
    ``observation.*`` features.  B4 needs future base pose/twist only as an
    auxiliary *label*, never as ACT or R0 input, so this type-gated extension
    adds the two explicitly allowed diagnostic keys and nothing else.
    """

    missing = [key for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS if key not in dataset_meta.features]
    if missing:
        raise ValueError(
            "B4 base auxiliary requires canonical train-label diagnostics "
            f"{list(BASE_AUXILIARY_DIAGNOSTIC_KEYS)}, dataset is missing {missing}"
        )
    result = dict(current or {})
    offsets = [0.0, float(config.base_auxiliary_horizon) / float(dataset_meta.fps)]
    for key in BASE_AUXILIARY_DIAGNOSTIC_KEYS:
        existing = result.get(key)
        if existing is not None and list(existing) != offsets:
            raise ValueError(
                f"B4 diagnostic window conflict for {key}: expected {offsets}, got {existing}"
            )
        result[key] = offsets
    return result


def make_reaction_aware_act_pre_post_processors(
    config: ReactionAwareACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    pretrained_path: str | None = None,
    **kwargs: Any,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse ACT's image/state/action processing without normalizing raw reaction state.

    ``observation.reaction_state`` is intentionally not placed in
    ``config.input_features``.  It remains a raw auxiliary observation whose
    normalizer lives inside the separately checkpointed reaction model; the
    standard ACT processor preserves unknown observation keys unchanged.
    """

    # A loaded pipeline serializes its steps but not the callable that converts
    # a raw batch into an EnvTransition.  Reattach our converter explicitly on
    # resume/inference; otherwise the generic loader would revert to
    # ``batch_to_transition`` and drop canonical ``diagnostics.*`` again.
    if pretrained_path is not None:
        if not config.reaction_aware:
            # Preserve the exact native ACT loader path when the plugin is
            # intentionally disabled.
            from lerobot.policies.factory import make_pre_post_processors

            return make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=pretrained_path,
                dataset_stats=dataset_stats,
                **kwargs,
            )
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=reaction_aware_batch_to_transition,
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

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )
    if config.reaction_aware:
        # Keep disabled plugin behavior byte-for-byte on ACT's processor path.
        preprocessor.to_transition = reaction_aware_batch_to_transition
    return preprocessor, postprocessor


def install_reaction_aware_act_processor_factory() -> None:
    """Teach upstream LeRobot's generic factory about this ACT subclass.

    LeRobot checks ``isinstance(config, ACTConfig)`` before its third-party
    dynamic-factory fallback, so an out-of-tree ACT subclass would otherwise be
    routed to the generic ACT preprocessor.  The shim is deliberately limited
    to :class:`ReactionAwareACTConfig`; every other policy keeps the upstream
    implementation.  Existing imports in ``lerobot_train`` are also rebound so
    plugin discovery performed inside its ``main()`` takes effect in the same
    process, including ``python -m lerobot.scripts.lerobot_train``.
    """

    import lerobot.datasets.factory as dataset_factory_module
    import lerobot.policies as policies_module
    import lerobot.policies.factory as factory_module

    current_resolver = dataset_factory_module.resolve_delta_timestamps
    if not getattr(current_resolver, "_spacepolicy_reaction_aware_act_resolver", False):

        @wraps(current_resolver)
        def reaction_aware_resolver(config: Any, dataset_meta: Any):
            resolved = current_resolver(config, dataset_meta)
            if (
                isinstance(config, ReactionAwareACTConfig)
                and config.reaction_aware
                and config.base_auxiliary_enabled
            ):
                return _reaction_aware_base_auxiliary_delta_timestamps(
                    resolved,
                    config=config,
                    dataset_meta=dataset_meta,
                )
            return resolved

        setattr(reaction_aware_resolver, "_spacepolicy_reaction_aware_act_resolver", True)
        dataset_factory_module.resolve_delta_timestamps = reaction_aware_resolver

    current_factory = factory_module.make_pre_post_processors
    if getattr(current_factory, "_spacepolicy_reaction_aware_act_factory", False):
        return

    @wraps(current_factory)
    def reaction_aware_factory(
        policy_cfg: Any,
        pretrained_path: str | None = None,
        **factory_kwargs: Any,
    ) -> tuple[Any, Any]:
        if isinstance(policy_cfg, ReactionAwareACTConfig) and policy_cfg.reaction_aware:
            return make_reaction_aware_act_pre_post_processors(
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

    setattr(reaction_aware_factory, "_spacepolicy_reaction_aware_act_factory", True)
    factory_module.make_pre_post_processors = reaction_aware_factory
    policies_module.make_pre_post_processors = reaction_aware_factory
    # ``lerobot_train`` imports the function at module import time.  By the
    # time it discovers third-party plugins, either its package module or the
    # ``-m`` ``__main__`` module is already live, so patch those bound globals.
    for module_name in ("lerobot.scripts.lerobot_train", "__main__"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "make_pre_post_processors", None) is current_factory:
            setattr(module, "make_pre_post_processors", reaction_aware_factory)


__all__ = [
    "make_reaction_aware_act_pre_post_processors",
    "reaction_aware_batch_to_transition",
    "install_reaction_aware_act_processor_factory",
]
