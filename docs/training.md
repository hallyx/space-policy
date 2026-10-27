# Training and deployment

All policies use LeRobot's native training loop. Install the package before
launching the trainer so LeRobot can discover the ACT extensions. Dataset,
optimizer, logging, resume, and checkpoint options use LeRobot's standard CLI.

## Policy settings

| Type | Chunk length | Actions per prediction | Micro-horizon | Additional requirements |
| --- | --- | --- | --- | --- |
| `act` | 32 | 8 | — | Standard LeRobot observations and actions |
| `spline_act` | 32 | 1 | 4 | State8 and action15 |
| `base_aux_spline_act` | 32 | 1 | 4 | Base-pose/twist label endpoints |
| `reaction_aware_act` | 32 | 1 | 4 | Raw reaction state, control-state gate, R0 checkpoint |

These values follow the included YAML profiles. Fields such as `state_dim`,
`action_dim`, and `action_contract` in those profiles describe the data schema;
they are not keyword arguments to the LeRobot config classes.

## Base-Aux ACT

The auxiliary variant expects the following additional dataset features:

| Field | Training batch shape | Semantics |
| --- | --- | --- |
| `diagnostics.base_pose` | `[B, 2, 7]` | World position3 + quaternion XYZW4 at t and t+K |
| `diagnostics.base_velocity` | `[B, 2, 6]` | World linear3 + angular3 velocity at t and t+K |
| `diagnostics.base_pose_is_pad` | `[B, 2]` | Endpoint padding mask |
| `diagnostics.base_velocity_is_pad` | `[B, 2]` | Endpoint padding mask |

The processor hooks request those endpoint windows using the dataset's FPS.
The pose/twist fields are loss labels, not visual ACT inputs. Derived targets
are relative SE(3) tangent6 and future body-frame twist6.

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=base_aux_spline_act \
  --dataset.repo_id=local/my_dataset \
  --dataset.root=/path/to/my_dataset \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=null \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=outputs/base_aux_act
```

`BaseAuxiliarySplineACTConfig` defaults to chunk_size=32,
micro_horizon=4, num_micro_chunks=8, and n_action_steps=1. Its auxiliary horizon
must equal the micro-horizon. Run the included `test_base_aux_spline_act_policy`
tests for synthetic forward/backward and checkpoint examples.

## Reaction ACT

The reaction model is an independently trained `LearnedReactionModel`. Supply
a compatible checkpoint through `reaction_model_checkpoint`; weights are not
included. Random transition weights used by unit tests are not a trained
reaction model.

The current implementation consumes either `observation.reaction_state`
`[B,80]`, or these current-state diagnostics:

- `diagnostics.base_pose`: 7 values, world position3 + quaternion XYZW4.
- `diagnostics.base_velocity`: 6 values, world linear3 + angular3 velocity.
- `diagnostics.raw_joint_state`: 26 measured articulation joint positions.
- `diagnostics.measured_joint_velocity`: 26 measured articulation velocities.

The preceding action15 completes the 80D reaction state. During training,
the configured action window includes the previous command plus the future
chunk. At an episode start, padded history is handled explicitly. When base
auxiliary supervision is enabled, base pose/twist retain their two-endpoint
label windows; only the current endpoint enters R0.

The preprocessor requires `diagnostics.control_state` in the permitted
pre-contact states 13, 14, or 15. This field gates admission and is not fed to
the network. This implementation does not provide a contact-phase fallback.

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=reaction_aware_act \
  --policy.reaction_aware=true \
  --policy.action_representation=hermite_joint_trajectory \
  --policy.chunk_size=32 \
  --policy.n_action_steps=1 \
  --policy.micro_horizon=4 \
  --policy.num_micro_chunks=8 \
  --policy.reaction_model_checkpoint=/path/to/reaction_model.pt \
  --policy.base_auxiliary_enabled=true \
  --policy.base_auxiliary_horizon=4 \
  --dataset.repo_id=local/my_dataset \
  --dataset.root=/path/to/my_dataset \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=null \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=10000 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=outputs/reaction_act
```

Setting `reaction_aware=true` explicitly is required: the config defaults to
the native ACT path when that flag is false. The pretrained reaction model
is frozen by default; gradients still pass through its candidate-action input
to the policy. Model and checkpoint contracts are implemented in
`modeling_reaction_aware_act.py` and exercised by the policy tests.

## Deployment

Use the saved configuration and pre/postprocessor files from the same
checkpoint. The processors perform normalization and device transfer. They do
not capture camera images, read robot state, or execute motor commands; those
interfaces belong to your deployment system.

For Reaction ACT, supply the correctly ordered raw reaction state on each tick
and retain the pre-contact admission field. For all variants, keep the
training control frequency and action units, and reset policy state when an
episode changes. Normalization statistics and weights are user-provided
artifacts, not part of this repository.
