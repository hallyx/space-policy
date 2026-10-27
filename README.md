# SpacePolicy

**ACT-based visuomotor learning for free-floating space manipulation.**

SpacePolicy extends [LeRobot](https://github.com/huggingface/lerobot)'s Action
Chunking Transformer (ACT) with base-motion supervision and learned reaction
modeling. Multi-view images and joint observations drive action-chunk
prediction, while trajectory decoding and dynamics-related objectives support
experiments with coupled base–arm motion.

This repository provides the ACT-based policies, configuration profiles, and
tests. Simulation scenes, robot and satellite assets, datasets, and pretrained
weights are not included.

## Methods

| Policy | Description |
| --- | --- |
| `act` | Native LeRobot ACT baseline with CVAE training and action chunking |
| `spline_act` | ACT proposals decoded into piecewise Hermite joint trajectories |
| `base_aux_spline_act` | Spline ACT with an action-conditioned base-motion auxiliary head |
| `reaction_aware_act` | ACT with learned reaction-state rollout and optional base auxiliary supervision |

The auxiliary head predicts relative base motion and future velocity for
training supervision. The Reaction model predicts robot-state transitions
between action micro-chunks. Hermite decoding, rigid-body geometry, state
contracts, and processor hooks support these ACT extensions.

Detailed architecture and manuscript correspondence are in
[Implementation notes](docs/implementation.md).

## Installation

Use Python 3.12. For GPU training, install PyTorch for your CUDA version first.

```bash
git clone https://github.com/hallyx/space-policy.git
cd space-policy
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The package is `lerobot_policy_spacepolicy_act` and targets LeRobot 0.5.1.
LeRobot discovers its policy configurations and processor hooks automatically
when the package is installed. You can also import it explicitly:

```python
import lerobot_policy_spacepolicy_act
from lerobot.policies.factory import get_policy_class

policy_class = get_policy_class("base_aux_spline_act")
```

If upgrading from the earlier package name, follow the
[compatibility notes](docs/implementation.md#package-compatibility).

## Data

Prepare your own LeRobot-format dataset with consistent camera names, joint
order, units, and control frequency.

| Feature | Shape per timestep | Meaning |
| --- | --- | --- |
| `observation.images.<camera>` | `[3, H, W]` | RGB input for each camera |
| `observation.state` | `[8]` | Seven measured arm positions and one gripper position |
| `action` | `[15]` | Seven commanded positions, seven desired velocities, one gripper command |

LeRobot handles dataset loading, normalization, temporal windows, and
checkpoints. The native ACT baseline can infer feature dimensions from dataset
metadata. The included extensions use the state8/action15 contract above.

Base-Aux ACT additionally needs base-pose and base-velocity label windows.
Reaction ACT needs the current raw reaction state, preceding command, and a
compatible transition-model checkpoint. See
[Training and data contracts](docs/training.md) for the required fields.

## Training

Train the native ACT baseline with LeRobot's standard entry point:

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=local/my_dataset \
  --dataset.root=/path/to/my_dataset \
  --policy.device=cuda \
  --policy.chunk_size=32 \
  --policy.n_action_steps=8 \
  --policy.pretrained_backbone_weights=null \
  --policy.optimizer_lr=1e-5 \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=10000 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=outputs/act
```

Replace the dataset identifier and path with your own, and use a new output
directory for each run. This command uses the ACT architecture defaults in
LeRobot 0.5.1 and the project's 32-step/8-action baseline settings.

The [policy profiles](configs/policies) describe all four variants. They
contain model settings and descriptive schema fields, rather than complete
serialized LeRobot training configurations. Extension commands and checkpoint
requirements are documented in the [training guide](docs/training.md).

## Inference

Load a checkpoint with its saved policy configuration and processors:

```python
import torch
import lerobot_policy_spacepolicy_act
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

checkpoint = "outputs/act/checkpoints/last/pretrained_model"
config = PreTrainedConfig.from_pretrained(checkpoint)
policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
preprocess, postprocess = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=checkpoint,
)
policy.eval()
policy.reset()

@torch.inference_mode()
def predict(observation):
    return postprocess(policy.select_action(preprocess(observation)))
```

Supply current RGB float CHW tensors in `[0, 1]` and the measured joint state,
using the same camera names and physical units as training. Execute the output
through your robot or simulator interface. Reset policy history at episode
boundaries. Reaction ACT also requires its raw-state inputs and pre-contact
admission fields, as described in the training guide.

Native ACT supports temporal aggregation with `n_action_steps=1` and a
`temporal_ensemble_coeff`. Spline, Base-Aux, and Reaction variants execute the
latest single action and disable ACT temporal aggregation.

## Repository layout

```text
configs/policies/                       ACT and extension profiles
src/lerobot_policy_spacepolicy_act/
  modeling_spline_act.py                Hermite action decoding
  modeling_base_aux_spline_act.py       ACT with base auxiliary supervision
  modeling_reaction_aware_act.py        ACT with reaction rollout
  reaction_model.py                    Learned state transition
  base_auxiliary.py                    Base-motion labels, head, and losses
  configuration_*.py                   LeRobot policy configurations
  processor_*.py                       Label and state processing hooks
  geometry.py                          Differentiable rigid-body geometry
  trajectory.py                        Hermite trajectory sampling
  reaction_state.py                    Raw reaction-state contracts
  reaction_phase_gate.py               Pre-contact admission checks
  act_temporal_ensemble.py              ACT inference aggregation
docs/                                  Training and implementation details
tests/                                 Policy, processor, and geometry tests
```

## Tests

```bash
OMP_NUM_THREADS=2 python -m pytest -q
```

Tests use synthetic tensors and temporary checkpoints. They cover action and
base losses, reaction rollout, processor boundaries, policy registration,
checkpoint round trips, temporal aggregation, and geometry. No scene or
dataset is required.

## Citation

If you use SpacePolicy in your research, please cite the paper below.

[Paper](https://rasevents.org/uploads/documents/pdfviewer/c1/0a/235289-1549.pdf) ·
[Download BibTeX](CITATION.bib)

```bibtex
@inproceedings{han2026spacepolicy,
  author    = {Han, Bing and Hu, Xing and Wang, Bingquan and Zhan, Runzhe and
               Yang, Jianing and Fu, Zhihang and Zhao, Xiangkang and
               Wang, Chenxi and Zhai, Zhi and Liu, Jinxin and Chen, Xuefeng},
  title     = {{SpacePolicy}: Dynamics-Aware Imitation Learning for Free-Floating On-Orbit Capture},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  pages     = {9912--9919},
  publisher = {IEEE},
  url       = {https://rasevents.org/uploads/documents/pdfviewer/c1/0a/235289-1549.pdf}
}
```

## Acknowledgements

SpacePolicy uses LeRobot's ACT backbone, CVAE, training loop, processors, and
checkpoint infrastructure. ACT was introduced by Zhao et al. in
*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.

## License

MIT. See [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).
