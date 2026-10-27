"""Out-of-tree SpacePolicy policies with lazy optional dependencies.

Importing a lightweight utility submodule (for example the legacy episode
sampler) must not eagerly import every experimental policy processor.  Those
processors track different LeRobot API versions and are loaded only when their
public symbol is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str, str | None]] = {'ReactionAwareACTConfig': ('.configuration_reaction_aware_act',
                            'ReactionAwareACTConfig',
                            'reaction'),
 'ReactionAwareACTPolicy': ('.modeling_reaction_aware_act', 'ReactionAwareACTPolicy', 'reaction'),
 'make_reaction_aware_act_pre_post_processors': ('.processor_reaction_aware_act',
                                                 'make_reaction_aware_act_pre_post_processors',
                                                 'reaction'),
 'install_reaction_aware_act_processor_factory': ('.processor_reaction_aware_act',
                                                  'install_reaction_aware_act_processor_factory',
                                                  'reaction'),
 'SplineACTConfig': ('.configuration_spline_act', 'SplineACTConfig', None),
 'SplineACTPolicy': ('.modeling_spline_act', 'SplineACTPolicy', None),
 'BaseAuxiliarySplineACTConfig': ('.configuration_base_aux_spline_act',
                                  'BaseAuxiliarySplineACTConfig',
                                  'base_aux'),
 'BaseAuxiliarySplineACTPolicy': ('.modeling_base_aux_spline_act',
                                  'BaseAuxiliarySplineACTPolicy',
                                  'base_aux'),
 'make_base_aux_spline_act_pre_post_processors': ('.processor_base_aux_spline_act',
                                                  'make_base_aux_spline_act_pre_post_processors',
                                                  'base_aux'),
 'install_base_aux_spline_act_runtime_shims': ('.processor_base_aux_spline_act',
                                               'install_base_aux_spline_act_runtime_shims',
                                               'base_aux')}

__all__ = list(_LAZY_EXPORTS)
_INSTALLED_HOOKS: set[str] = set()


def _install_hook(hook: str | None) -> None:
    if hook is None or hook in _INSTALLED_HOOKS:
        return
    if hook == "reaction":
        processor = import_module(".processor_reaction_aware_act", __name__)
        processor.install_reaction_aware_act_processor_factory()
    elif hook == "base_aux":
        _install_hook("reaction")
        processor = import_module(".processor_base_aux_spline_act", __name__)
        processor.install_base_aux_spline_act_runtime_shims()
    else:  # pragma: no cover - guarded by the static export table.
        raise ValueError(f"unknown SpacePolicy import hook: {hook}")
    _INSTALLED_HOOKS.add(hook)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name, hook = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    _install_hook(hook)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
