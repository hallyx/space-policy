from __future__ import annotations

import torch

from lerobot_policy_spacepolicy_act.trajectory import (
    evaluate_arm7_cubic_hermite,
    sample_arm7_cubic_hermite,
)


def test_arm7_hermite_hits_position_and_velocity_boundaries() -> None:
    start_position = torch.linspace(-0.4, 0.2, 7, dtype=torch.float64)
    start_velocity = torch.linspace(0.1, -0.2, 7, dtype=torch.float64)
    end_position = torch.linspace(0.7, -0.3, 7, dtype=torch.float64)
    end_velocity = torch.linspace(-0.05, 0.15, 7, dtype=torch.float64)
    positions, velocities = evaluate_arm7_cubic_hermite(
        start_position,
        start_velocity,
        end_position,
        end_velocity,
        torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
        duration=0.2,
    )

    assert positions.shape == (3, 7)
    assert velocities.shape == (3, 7)
    torch.testing.assert_close(positions[0], start_position, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(velocities[0], start_velocity, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(positions[-1], end_position, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(velocities[-1], end_velocity, atol=1.0e-12, rtol=0.0)


def test_adjacent_arm7_hermite_segments_are_c1_continuous() -> None:
    start = torch.zeros(7, dtype=torch.float64)
    shared_position = torch.linspace(0.1, 0.7, 7, dtype=torch.float64)
    end = torch.linspace(-0.2, 0.4, 7, dtype=torch.float64)
    start_velocity = torch.linspace(-0.2, 0.1, 7, dtype=torch.float64)
    shared_velocity = torch.linspace(0.05, -0.1, 7, dtype=torch.float64)
    end_velocity = torch.linspace(0.2, -0.05, 7, dtype=torch.float64)
    first_position, first_velocity = evaluate_arm7_cubic_hermite(
        start, start_velocity, shared_position, shared_velocity, 0.3, duration=0.3
    )
    second_position, second_velocity = evaluate_arm7_cubic_hermite(
        shared_position, shared_velocity, end, end_velocity, 0.0, duration=0.3
    )

    torch.testing.assert_close(first_position, second_position, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(first_velocity, second_velocity, atol=1.0e-12, rtol=0.0)


def test_arm7_hermite_supports_batches_and_uniform_sampling() -> None:
    start = torch.zeros(2, 7, dtype=torch.float64)
    start_velocity = torch.full((2, 7), 0.1, dtype=torch.float64)
    end = torch.stack((torch.ones(7, dtype=torch.float64), -torch.ones(7, dtype=torch.float64)))
    end_velocity = torch.full((2, 7), -0.2, dtype=torch.float64)
    duration = torch.tensor([0.2, 0.4], dtype=torch.float64)
    positions, velocities = evaluate_arm7_cubic_hermite(
        start,
        start_velocity,
        end,
        end_velocity,
        torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
        duration=duration,
    )

    assert positions.shape == (2, 3, 7)
    assert velocities.shape == (2, 3, 7)
    torch.testing.assert_close(positions[:, 0], start, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(positions[0, -1], end[0], atol=1.0e-12, rtol=0.0)
    sampled = sample_arm7_cubic_hermite(
        start,
        start_velocity,
        end,
        end_velocity,
        duration=duration,
        num_samples=5,
    )
    assert sampled.positions.shape == (2, 5, 7)
    assert sampled.velocities.shape == (2, 5, 7)
    assert sampled.sample_times.shape == (2, 5)
    torch.testing.assert_close(sampled.positions[:, 0], start, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(sampled.positions[:, -1], end, atol=1.0e-12, rtol=0.0)


def test_arm7_hermite_propagates_finite_gradients() -> None:
    start = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    start_velocity = torch.full((7,), 0.1, dtype=torch.float64, requires_grad=True)
    end = torch.linspace(0.2, 0.8, 7, dtype=torch.float64, requires_grad=True)
    end_velocity = torch.full((7,), -0.15, dtype=torch.float64, requires_grad=True)
    duration = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    positions, velocities = evaluate_arm7_cubic_hermite(
        start,
        start_velocity,
        end,
        end_velocity,
        torch.tensor([0.05, 0.2], dtype=torch.float64),
        duration=duration,
    )
    (positions.square().sum() + velocities.square().sum()).backward()

    for value in (start, start_velocity, end, end_velocity, duration):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
