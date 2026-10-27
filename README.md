# SpacePolicy — Existing Method Source

This release extracts existing ACT-related method code from the SpacePolicy
project. It does **not** introduce a new policy implementation or claim that the
available code fully implements the supplied IROS manuscript. Read
[PAPER_CODE_AUDIT.md](PAPER_CODE_AUDIT.md) before treating it as a reproduction.

## Included original implementations

| Component | Original policy / module | Relationship to the manuscript |
| --- | --- | --- |
| ACT backbone and CVAE | Upstream LeRobot dependency | Shared backbone; not vendored |
| Joint-PV action15 | `modeling_spline_act.py`, four original policy profiles | Matches position7 + velocity7 + gripper1 layout |
| B2 Spline ACT | `SplineACTPolicy` | Existing Hermite trajectory variant; not the paper's token-conditioned decoder |
| B3 base auxiliary | `BaseAuxiliarySplineACTPolicy`, `base_auxiliary.py` | Closest base-prediction component, but train-only and action-conditioned |
| B4 Reaction-Aware ACT | `ReactionAwareACTPolicy`, `reaction_model.py` | Related predicted-state rollout; different architecture and runtime inputs |
| Temporal aggregation | `act_temporal_ensemble.py` | Existing ACT inference utility; disabled by B2/B3/B4 profiles |
| Geometry / label / processor helpers | Transitive source dependencies | Required to run these original components |

16 original method/dependency modules, 4 original policy profiles, 10 original
test files, and the existing MIT license are copied byte for byte. The original
package `__init__.py` retains its lazy loader and runtime hooks; only its export
table is narrowed to the included methods. No model mathematics, losses,
parameter defaults, or control behavior were rewritten.

[provenance.json](provenance.json) records original relative paths, hashes,
working-tree identity, and the one packaging edit.
[packaging_changes.patch](packaging_changes.patch) shows that edit.

## Release boundary

Only method source, its tests, generic policy profiles, and documentation are
included. There are no scenes, robot/satellite assets, datasets, episode lists,
calibrations, learned normalization statistics, checkpoints, experiment results,
PDFs, recordings, or simulator clients. The prior newly written reference
implementation is superseded and excluded. DiT placeholders and TargetVision
experiments are not presented as this manuscript's method.

The manuscript's hindsight data-construction idea is documented in the audit;
the collector is coupled to scene/simulation code and is not bundled. The
full `M -> future-base-sequence -> physics-token -> action-decoder` path was
not located in the inspected current source trees. This release preserves
that gap instead of silently filling it with new model code.

## Installation and checks

Use a separate Python 3.12 environment for this package. It retains the original
package namespace and original dependency pins:

```bash
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=2 python -m pytest -q
python tools/verify_provenance.py
```

The 55 extracted original tests passed in the existing local environment with
LeRobot 0.5.2 from a local source checkout; the project's metadata pins 0.5.1.
A fresh installation of those original pins has not been verified. The release
does not silently change them. See [VALIDATION.md](VALIDATION.md).

To check that copied files still match an available source workspace:

```bash
python tools/verify_provenance.py --source-root /path/to/original/space_policy
```

The four YAMLs are original descriptive profiles, not directly serialized
LeRobot configs. Use the actual config classes and set LeRobot `input_features`
and `output_features` to your own compatible observations and action15.

## Run an original component

The existing tests provide small synthetic, CPU-only examples without a scene
or dataset. For example:

```bash
python -m pytest tests/test_base_aux_spline_act_policy.py -q
python -m pytest tests/test_reaction_aware_act_policy.py -q
```

B3 expects current RGB and `observation.state` with 8 values, 15D action chunks,
and two train-label windows: `diagnostics.base_pose` `[B,2,7]` and
`diagnostics.base_velocity` `[B,2,6]`, plus padding masks. Labels are the endpoints
at t and t+micro_horizon, not a K-step 9D absolute pose sequence. Inference
needs only its ordinary RGB/state inputs and training-derived processors.

B4 additionally consumes the current raw reaction state (80 values) and uses a
separately trained transition model. That input boundary is different from the
manuscript; no trained transition weights are supplied. A random model in a
unit test is only a wiring check. The original pre-contact phase gate remains
in place and is not a validated full-capture controller.

Training and inference infrastructure are supplied by LeRobot; this release
does not ship environment-specific launchers. A frozen `reaction_model` must
be trained and validated by the user before meaningful B4 use. For the exact
API, follow the included config classes, processors, and existing tests rather
than the superseded reference package.

## License and attribution

MIT, retained from the original repository. ACT implementations are imported
from LeRobot rather than copied. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This package does not assert
that its components produced the manuscript's tables or establish the
manuscript's publication/acceptance status.
