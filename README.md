# SpacePolicy

**ACT-based visuomotor learning for free-floating space manipulation.**

SpacePolicy builds on [LeRobot](https://github.com/huggingface/lerobot)'s Action
Chunking Transformer (ACT) to study robotic manipulation under base–arm
coupling. It provides joint position–velocity action prediction, trajectory
parameterization, auxiliary base-state prediction, and learned reaction
modeling within the LeRobot policy interface.

The repository contains the ACT-based method components, policy configurations,
and tests. Simulation scenes, robot and satellite assets, datasets, and trained
weights are not distributed.

## Features

- **LeRobot ACT backbone:** multi-view RGB and proprioceptive inputs, CVAE
  training, action-chunk prediction, and native checkpoint support.
- **Joint position–velocity actions:** seven arm positions, seven desired
  velocities, and one gripper command.
- **Spline trajectory decoding:** piecewise cubic Hermite trajectories from
  ACT action proposals.
- **Base auxiliary prediction:** action-conditioned supervision of relative
  base motion and future base velocity.
- **Reaction-aware planning:** a learned transition model propagates predicted
  robot state across action micro-chunks.
- **ACT temporal aggregation:** utilities for combining overlapping action
  predictions in the native ACT inference path.

## Policy variants

| Policy type | Implementation | Description |
| --- | --- | --- |
| `act` | LeRobot `ACTPolicy` | Native visual action-chunking baseline |
| `spline_act` | `SplineACTPolicy` | ACT with Hermite joint-trajectory decoding |
| `base_aux_spline_act` | `BaseAuxiliarySplineACTPolicy` | Spline ACT with a base-motion auxiliary objective |
| `reaction_aware_act` | `ReactionAwareACTPolicy` | ACT with action-conditioned reaction rollout and optional base auxiliary supervision |

The base auxiliary head in `base_aux_spline_act` is used for training
supervision. In `reaction_aware_act`, the separate reaction model predicts
states used by subsequent micro-chunks. These paths are implemented in the
existing project code and retain their original behavior.

## Installation

Use Python 3.12 and a PyTorch installation compatible with your hardware.

```bash
git clone https://github.com/hallyx/space-policy.git
cd space-policy
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Dependencies are declared in `pyproject.toml`. The Python package name
`lerobot_policy_spacepolicy_dit` is retained for compatibility with the existing
LeRobot plugin and checkpoints; the policy implementations included here are
ACT-based.

## Data interface

Provide your own dataset in LeRobot format. The training examples use the
following observation and action layout:

| Field | Shape | Description |
| --- | --- | --- |
| `observation.images.<camera>` | `[3, H, W]` | RGB image for each camera view |
| `observation.state` | `[8]` | Seven arm joint positions and one gripper position |
| `action` | `[15]` | Arm positions7 + desired velocities7 + gripper command1 |

Camera names and feature dimensions are read from dataset metadata. Keep
joint ordering, units, camera order, and sampling frequency consistent between
training and deployment. LeRobot handles dataset loading and normalization.

The base auxiliary variant additionally requires base-pose and base-velocity
label windows. Reaction-aware policies require the current reaction state and
a compatible learned transition model. Their exact fields and padding
contracts are defined in the corresponding config and processor modules.

## Training

The native ACT baseline uses LeRobot's training entry point. Replace the
dataset identifier and local path with your own:

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=local/my_dataset \
  --dataset.root=/path/to/my_dataset \
  --policy.device=cuda \
  --policy.chunk_size=32 \
  --policy.n_action_steps=8 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=null \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.optimizer_lr=1e-5 \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=10000 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=outputs/act
```

The settings above follow the project's native ACT baseline. Use a new output
directory for each run. Policy profiles for all four variants are in
[`configs/policies`](configs/policies); they document the model and execution
settings. The YAML profiles include descriptive fields such as `state_dim`
and `action_contract` and are not complete LeRobot training config files.

For the ACT extensions, use the registered policy type and its matching config
class, processor, and required dataset fields. Reaction-aware training uses a
separate transition-model checkpoint through `reaction_model_checkpoint`.
See the included policy tests for small, executable examples of constructing,
training, and saving each variant without a simulator.

## Inference

Load a trained ACT checkpoint with LeRobot:

```python
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

checkpoint = "outputs/act/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(checkpoint)
preprocess, postprocess = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=checkpoint,
)
policy.eval()
policy.reset()

@torch.inference_mode()
def predict(observation):
    # Current RGB float CHW tensors in [0, 1], plus the measured joint state.
    return postprocess(policy.select_action(preprocess(observation)))
```

Use the checkpoint's saved processors and statistics. Execute the returned
action through your own robot or simulator interface, then acquire the next
observation. Reset policy history at every episode boundary.

For native ACT temporal aggregation, set `n_action_steps=1` and a
`temporal_ensemble_coeff`, for example `0.01`. The spline and reaction variants
use latest-plan, single-step execution and disable ACT temporal aggregation.
The current reaction implementation is restricted to its pre-contact state
contract; see the phase gate and processor for the supported inputs.

## Project structure

```text
configs/policies/                 Policy profiles
src/lerobot_policy_spacepolicy_dit/
  modeling_spline_act.py          ACT trajectory decoder
  modeling_base_aux_spline_act.py ACT with base auxiliary supervision
  modeling_reaction_aware_act.py  ACT with reaction rollout
  reaction_model.py              Learned dynamics transition model
  base_auxiliary.py              Base-motion prediction and losses
  processor_*.py                 LeRobot data processors
  configuration_*.py             Registered policy configurations
  geometry.py                    Rigid-body geometry utilities
  trajectory.py                  Hermite trajectory sampling
  act_temporal_ensemble.py        ACT inference aggregation
tests/                           Policy, geometry, and processor tests
```

## Tests

```bash
OMP_NUM_THREADS=2 python -m pytest -q
```

The tests use synthetic tensors and temporary checkpoints; no dataset or scene
is required. Environment details and verification scope are recorded in
[`VALIDATION.md`](VALIDATION.md). Implementation correspondence with the
manuscript is documented separately in
[`PAPER_CODE_AUDIT.md`](PAPER_CODE_AUDIT.md).

## Acknowledgements

This project uses LeRobot's ACT backbone, training loop, processors, and
checkpoint infrastructure. ACT was introduced in *Learning Fine-Grained
Bimanual Manipulation with Low-Cost Hardware* by Zhao et al.

## License

MIT. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependency attribution.
