# Third-party notices

SpacePolicy's ACT extensions, configurations and utilities are distributed
under the repository's MIT license. The project reuses its existing method
implementations; it does not vendor the upstream ACT model.

LeRobot supplies ACT, the CVAE, standard processors, training and checkpoint
infrastructure under Apache-2.0:
https://github.com/huggingface/lerobot/blob/main/LICENSE

ACT was introduced by Zhao et al. in *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware*:
https://github.com/tonyzhaozh/act

PyTorch, torchvision, NumPy, Transformers, and other transitive dependencies
retain their own licenses. LeRobot 0.5.1 also declares Diffusers as a base
dependency; its presence in an installed environment does not mean this
repository includes a diffusion-policy implementation.

Simulation scenes, datasets, calibration files and pretrained weights are not
included. Project-page media and the paper figure retain their respective
rights; see `docs/assets/README.md`. They are not covered by the MIT code license.
