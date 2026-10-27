# Paper / existing-code correspondence audit

Scope: the supplied manuscript *SpacePolicy: Dynamics-Aware Imitation Learning
for Free-Floating On-Orbit Capture*, Section III-B, Figure 3, Algorithms 2–3,
and the Section IV-E ablations. Source inspection covered the active
`space_policy` working tree, the other discovered `space_policy` source tree,
and relevant collector routines. It is not an exhaustive search of all
external machines, historical checkpoints, or Git history.

The six compared core files (B2/B3/B4 models, base auxiliary, and the two DiT
placeholder modules) were byte-identical across the two inspected source trees.
No complete implementation of the manuscript's token-conditioning path was
located in those trees. Existing related components are released without
renaming them to imply architectural equivalence.

## Detailed mapping

| Item | Manuscript | Existing code and evidence | Assessment |
| --- | --- | --- | --- |
| Backbone | ACT encoder/decoder and CVAE | B2 inherits LeRobot `ACTPolicy`; B3 inherits B2; B4 inherits `ACTPolicy` | Shared backbone; upstream behavior rather than a new ACT implementation |
| Action | Eq. (5), q7 + dq7 + gripper1 | B2 feature validation and profiles require action15, state8 | Matching layout |
| Base predictor input | g_psi predicts future base poses from encoder memory M | `base_auxiliary.py:176`, `ActionConditionedBaseAuxiliaryHead` takes emitted action micro-chunks; B3 constructs this after decoding | Different dependency direction |
| Base output | A horizon of poses represented by p3 + rotation6d6, i.e. 9 scalar values/step | B3 emits one relative SE(3) tangent6 and a future body-twist6; `base_auxiliary.py:101` derives two-endpoint labels | Different target, frame, dimensionality, and temporal coverage |
| Physics conditioning | Embed the predicted base sequence into c_b; cross-attend during action decoding | `modeling_base_aux_spline_act.py:114` evaluates a train-only output head; its docstring explicitly prohibits an edge back to action decoding | Missing from B3; not a paper-equivalent auxiliary ablation |
| Predicted-state feedback | Paper uses M/c_b/z action cross-attention | `modeling_reaction_aware_act.py:607` rolls R0 over Hermite micro-chunks and feeds `next_robot_state` into later segments | B4 provides a different state-rollout mechanism; not the Figure 3 architecture |
| Runtime observations | Multi-view RGB and proprioception; base future is predicted | B3 uses RGB/state8, but B4 constructs raw reaction state from base pose/twist, q26/dq26, previous action15 (80D) | B3 observation boundary is close; B4 requires additional current dynamics inputs |
| Objective | Algorithm 2: action MSE + lambda base MSE + beta KL | B3/B4 use masked action L1; B3 adds knot/initial-velocity loss; base objective uses translation MSE, quaternion-geodesic rotation loss, twist MSE | Different losses; preserved without modification |
| Horizon | k=100 default, k=20 ablation | Supplied ACT/B2/B3/B4 profiles use `chunk_size=32`; B3 uses `base_auxiliary_horizon=4` | Different configuration; not changed to fabricate a paper run |
| Aggregation | Algorithm 3 maintains overlapping chunks with exp(-m*i) weights | `act_temporal_ensemble.py:18` implements bounded ACT aggregation; B2/B3/B4 reject temporal ensembling and use latest-plan single-step execution | Utility exists; not wired/enabled in these released variants |
| Full capture | Approach–Align–Capture semantic trajectory, contact evaluation | B4 config and processor seal a pre-contact phase gate | No full-capture or table-performance equivalence established |

Line references above use the unmodified method files in this release. The
package initializer is the sole edited original Python file.

## DiT placeholders are not evidence of a complete method

In both inspected source trees, `modules/physics_conditioner.py` is an identity
mapping; `modules/future_base_decoder.py` is a linear projection taking an
`action_dim`, and its docstring describes future actions. The root configs
`full_spacepolicy.yaml` and `aux_base.yaml` say these physics/future-base paths
are reserved and not wired. The DiT model wraps MultiTaskDiT rather than the
ACT architecture specified by this manuscript. These placeholder modules and
profiles are excluded from the method package to avoid suggesting they are the
missing g_psi/c_b implementation.

## Hindsight data construction

The external collector's `DirectVLACollector.py` contains
`correct_satellite_pose_from_saved` and `_correct_satellite_using_real_pose`.
They compute a corrected satellite pose with tool-offset/frame corrections,
using `T_keypoint @ T_fix @ T_ks` or `T_ee @ T_fix @ T_ks`.
This is related to the manuscript's terminal-pose hindsight idea, but is not
identical to simply assuming coincident end-effector/keypoint frames.

One inspected call to `correct_satellite_pose_from_saved` is commented out;
its presence as a function does not prove that the currently used rollout
executes it. Collector/replay code also depends on simulator scenes and
calibrations. This audit does not establish which collection entrypoint
produced the paper's demonstrations or certify end-to-end pixel/action
alignment. The collector, scene configuration, and demonstration outputs are
excluded from the release. No synthetic replacement was written.

## Extraction and edits

- 16 original method/dependency modules copied verbatim.
- 4 original policy profiles and 10 original test files copied verbatim.
- Existing license copied verbatim.
- Original `__init__.py`: export table narrowed to the included methods;
  loader and registration hooks retained. See `packaging_changes.patch`.
- New files are packaging metadata, documentation, source-verification tooling,
  and the separate website. No new model, base predictor, or inference
  algorithm has been introduced.
- Per-file hashes identify the current working-tree snapshot, including
  pre-existing uncommitted content. The HEAD commit alone is not sufficient
  to identify this export.

## Interpretation

This is a traceable release candidate for the available, related method code.
The action representation and ACT foundations can be reused. The available
base-prediction and reaction variants must be described under their actual
B2/B3/B4 names. A complete paper-matching release still requires locating the
original g_psi/c_b implementation and its experiment configuration; those gaps
are reported explicitly rather than filled by newly authored reference code.
No performance result in the manuscript is claimed as validated by these tests.
