from __future__ import annotations

import torch

from lerobot_policy_spacepolicy_act.reaction_model import (
    LEFT_ARM_INDICES,
    LearnedReactionModel,
    REACTION_STATE_DIM,
    reaction_base_transition_loss,
    reaction_transition_loss,
    rollout_reaction_model,
)


def _state(batch_size: int = 2) -> torch.Tensor:
    state = torch.zeros(batch_size, REACTION_STATE_DIM)
    state[:, 6] = 1.0  # canonical XYZW identity quaternion.
    return state


def test_reaction_transition_is_explicitly_conditioned_on_candidate_action() -> None:
    torch.manual_seed(0)
    model = LearnedReactionModel(hidden_dim=24, num_layers=2)
    state = _state(1)
    stationary = torch.zeros(1, 2, 15)
    candidate = stationary.clone()
    candidate[:, -1, :7] = 0.25
    candidate[:, -1, 7:14] = -0.5

    first = model(state, stationary)
    second = model(state, candidate)

    assert not torch.allclose(first.next_robot_state, second.next_robot_state)
    arm_index = torch.tensor(LEFT_ARM_INDICES)
    # The learned residual may model servo tracking lag, but the final command is
    # the structural baseline for the directly actuated arm.
    assert torch.all(
        second.next_robot_state[:, 13:39][:, arm_index]
        > first.next_robot_state[:, 13:39][:, arm_index]
    )
    assert torch.all(
        second.next_robot_state[:, 39:65][:, arm_index]
        < first.next_robot_state[:, 39:65][:, arm_index]
    )


def test_reaction_model_returns_a_valid_se3_pose_without_task_or_visual_input() -> None:
    model = LearnedReactionModel(hidden_dim=16, num_layers=1, predict_uncertainty=True)
    prediction = model(_state(), torch.randn(2, 4, 15))

    assert prediction.next_robot_state.shape == (2, REACTION_STATE_DIM)
    assert prediction.body_delta_se3.shape == (2, 6)
    assert prediction.uncertainty_log_variance is not None
    assert torch.isfinite(prediction.next_robot_state).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.next_base_pose_world_xyzw[:, 3:], dim=-1),
        torch.ones(2),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_reaction_rollout_is_batched_differentiable_and_does_not_need_future_state() -> None:
    torch.manual_seed(1)
    model = LearnedReactionModel(hidden_dim=16, num_layers=2)
    initial = _state().requires_grad_()
    trajectory = torch.randn(2, 3, 2, 15, requires_grad=True)

    states, predictions = rollout_reaction_model(model, initial, trajectory)
    assert states.shape == (2, 4, REACTION_STATE_DIM)
    assert len(predictions) == 3
    loss = states.square().mean()
    loss.backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert trajectory.grad is not None and torch.isfinite(trajectory.grad).all()


def test_reaction_transition_loss_has_pose_twist_and_joint_terms() -> None:
    model = LearnedReactionModel(hidden_dim=16, num_layers=1)
    target = _state()
    prediction = model(_state(), torch.zeros(2, 1, 15))
    loss = reaction_transition_loss(prediction, target)

    assert torch.isfinite(loss.total)
    assert all(torch.isfinite(value) for value in (loss.position, loss.rotation, loss.twist, loss.joint))
    loss.total.backward()


def test_reaction_base_transition_loss_consumes_only_transition_labels() -> None:
    model = LearnedReactionModel(hidden_dim=16, num_layers=1)
    prediction = model(_state(), torch.zeros(2, 4, 15))
    target_delta = torch.zeros(2, 7)
    target_delta[:, 6] = 1.0
    target_twist_body = torch.zeros(2, 6)

    loss = reaction_base_transition_loss(
        prediction,
        target_delta_pose_body_xyzw=target_delta,
        target_future_twist_body=target_twist_body,
    )
    assert torch.isfinite(loss.total)
    assert all(torch.isfinite(value) for value in (loss.translation, loss.rotation, loss.twist))
    loss.total.backward()


def test_reaction_model_can_overfit_a_tiny_nonprivileged_transition_batch() -> None:
    """Exercise the actual R0 optimization signal, not just a single backward call."""

    torch.manual_seed(23)
    model = LearnedReactionModel(hidden_dim=16, num_layers=1)
    state = _state()
    state[:, 13:39] = 0.1
    candidate = torch.randn(2, 2, 15) * 0.1
    target_delta = torch.tensor(
        [
            [0.15, -0.10, 0.05, 0.0, 0.0, 0.0998334, 0.9950042],
            [-0.10, 0.05, 0.03, 0.0, 0.0499792, 0.0, 0.9987503],
        ],
        dtype=torch.float32,
    )
    target_twist = torch.tensor(
        [[0.20, -0.10, 0.05, 0.01, 0.02, -0.03], [-0.10, 0.20, -0.05, 0.03, -0.02, 0.01]],
        dtype=torch.float32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-3)

    def compute_loss() -> torch.Tensor:
        prediction = model(state, candidate)
        return reaction_base_transition_loss(
            prediction,
            target_delta_pose_body_xyzw=target_delta,
            target_future_twist_body=target_twist,
        ).total

    initial = float(compute_loss().detach())
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        optimizer.step()
    final = float(compute_loss().detach())

    assert final < initial * 0.01
