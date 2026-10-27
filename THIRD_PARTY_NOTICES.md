# Attribution and dependencies

Method and test source files were extracted from the user's existing SpacePolicy
project under its existing MIT license. Their paths and hashes are recorded
in provenance.json. This is not a newly authored replacement model.

The ACT architecture, VAE, standard processors, optimizer/training loop,
checkpoint API, and policy infrastructure are imported from LeRobot. LeRobot
is not vendored here and retains its own Apache-2.0 license:
https://github.com/huggingface/lerobot/blob/main/LICENSE

ACT conceptual source: Zhao et al., Learning Fine-Grained Bimanual Manipulation
with Low-Cost Hardware. No standalone ACT repository source was copied.

PyTorch, torchvision, Transformers, Diffusers, NumPy, PyZMQ and other transitive
dependencies retain their own licenses. Scene assets, datasets and weights are
not included or relicensed by this package.
