# Implementation details

SpacePolicy uses native LeRobot ACT as its visual action proposal backbone.
The included extensions implement the following paths:

```text
RGB + joint state -> ACT -> joint-PV action chunk
                          -> Hermite trajectory decoding          [Spline ACT]
                          -> action-conditioned base auxiliary    [Base-Aux ACT]
                          -> micro-chunks + reaction-state rollout [Reaction ACT]
```

Spline decoding is a dependency of Base-Aux ACT and is also used by Reaction
ACT. Geometry, base-state projection, and phase admission helpers support
those methods and remain part of the package.

## Actions and temporal alignment

All included extension profiles use current measured state8 and commanded
action15: arm positions7, desired velocities7, and gripper position1.
The default profiles predict 32 steps. Spline, base-auxiliary and reaction
variants use a four-step micro-horizon and execute the newest single action
at each control tick. These variants do not use ACT temporal aggregation.

Joint angles and gripper joint positions are in radians; commanded arm
velocities are in radians/second. The profile control interval is 0.02 seconds.
The dataset and runtime must agree on joint order, camera order, rate, and
whether the first action label is the command at the observation timestep.

## Base auxiliary supervision

`BaseAuxiliarySplineACTPolicy` predicts a relative base SE(3) tangent (6 values)
and a future base-frame twist (6 values) from the generated action prefix.
It uses only a training auxiliary loss. Its prediction is not fed back into
the action decoder during inference.

Training labels are derived from `diagnostics.base_pose` at t and t+K,
encoded as position3 + quaternion XYZW4, and `diagnostics.base_velocity`
at the same two endpoints, encoded as world-frame linear3 + angular3.
The processor retains these windows and padding masks for loss computation;
they are excluded from the visual ACT input dictionary.

## Reaction rollout

`ReactionAwareACTPolicy` uses `LearnedReactionModel` to propagate a candidate
action micro-chunk into the state for the next micro-chunk. It consumes
80 raw values: base pose7, world twist6, measured full-articulation q26/dq26,
and the previous action15. The separately trained reaction model may be
frozen while optimizing the ACT policy. No reaction weights are bundled.

The current implementation enforces pre-contact control states 13, 14 and 15.
This gate is an admission condition, not a policy feature or a validated
contact-phase controller. Training also needs the preceding action; its
requested action window therefore includes the previous command plus the
future chunk. See `configuration_reaction_aware_act.py` and its processor.

## Relationship to the paper

The [SpacePolicy paper](https://rasevents.org/uploads/documents/pdfviewer/c1/0a/235289-1549.pdf)
introduces hindsight demonstration construction and future-base-conditioned
action learning for free-floating on-orbit capture. This release contains the
project's ACT-based action representation, base auxiliary supervision, and
reaction-model components.

The implementation below differs from the paper's proposed encoder-memory ->
future absolute base-pose sequence -> physics-token -> action-decoder path.
The available auxiliary head is action-conditioned and output-only; the
reaction variant supplies a separate state rollout. Their losses, labels,
and 32-step default profiles also differ from the paper's description.
This repository does not claim complete reproduction of the paper's
architecture or numerical tables.

## Package compatibility

The Python distribution and import namespace are now
`lerobot_policy_spacepolicy_act`. The registered policy type strings remain
`spline_act`, `base_aux_spline_act`, and `reaction_aware_act`. Model classes,
state-dict keys, model mathematics, and profile values are unchanged.

When upgrading an environment that has the earlier
`lerobot_policy_spacepolicy_dit` distribution installed, uninstall that old
package before installing this one. Use the new import path in Python code.
LeRobot configuration JSON and safetensors checkpoints use the same policy
type strings and state dictionaries. Python-pickled objects referring to the
old module namespace require migration; that format is not maintained here.
