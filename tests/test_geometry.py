from __future__ import annotations

import math

import torch

from lerobot_policy_spacepolicy_dit.geometry import (
    apply_body_delta_xyzw,
    pose_geodesic_interpolate_xyzw,
    quaternion_geodesic_distance_xyzw,
    quaternion_to_matrix_xyzw,
    relative_pose_body_xyzw,
    se3_exp_xyzw,
    se3_log_xyzw,
    so3_exp_xyzw,
)


def _identity_quaternion(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)


def test_body_relative_update_respects_canonical_world_from_base_contract() -> None:
    half_turn = math.sqrt(0.5)
    start_world_from_base = torch.tensor(
        [1.0, 2.0, 3.0, 0.0, 0.0, half_turn, half_turn], dtype=torch.float64
    )
    # One metre along the *current base* x axis: z+90 maps it to world y.
    body_delta = torch.cat(
        (torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), _identity_quaternion())
    )

    end_world_from_base = apply_body_delta_xyzw(start_world_from_base, body_delta)
    torch.testing.assert_close(
        end_world_from_base[:3], torch.tensor([1.0, 3.0, 3.0], dtype=torch.float64)
    )
    recovered_delta = relative_pose_body_xyzw(start_world_from_base, end_world_from_base)
    torch.testing.assert_close(recovered_delta[:3], body_delta[:3], atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        quaternion_geodesic_distance_xyzw(recovered_delta[3:], body_delta[3:]),
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_batched_rotations_are_proper_and_geodesic_is_sign_invariant() -> None:
    rotation_vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.5, 0.7], [0.0, 0.0, math.pi / 2.0]],
        dtype=torch.float64,
    )
    quaternion = so3_exp_xyzw(rotation_vectors)
    rotation = quaternion_to_matrix_xyzw(quaternion)
    identity = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(3, dtype=torch.float64), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        quaternion_geodesic_distance_xyzw(quaternion, -quaternion),
        torch.zeros(3, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        quaternion_geodesic_distance_xyzw(_identity_quaternion(), quaternion[-1]),
        torch.tensor(math.pi / 2.0, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_se3_exp_log_and_geodesic_endpoints_are_batched() -> None:
    tangent = torch.tensor(
        [[0.3, -0.1, 0.2, 0.1, -0.2, 0.3], [-0.2, 0.4, 0.1, -0.3, 0.1, 0.2]],
        dtype=torch.float64,
    )
    delta = se3_exp_xyzw(tangent)
    torch.testing.assert_close(se3_log_xyzw(delta), tangent, atol=1.0e-11, rtol=1.0e-11)

    start = torch.cat(
        (torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64), _identity_quaternion())
    )
    end = apply_body_delta_xyzw(start, delta[0])
    interpolated = pose_geodesic_interpolate_xyzw(start, end, torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(interpolated[0, :3], start[:3], atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(interpolated[1, :3], end[:3], atol=1.0e-11, rtol=0.0)
    torch.testing.assert_close(
        quaternion_geodesic_distance_xyzw(interpolated[0, 3:], start[3:]),
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        quaternion_geodesic_distance_xyzw(interpolated[1, 3:], end[3:]),
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-11,
        rtol=0.0,
    )


def test_geometry_operations_propagate_finite_gradients() -> None:
    tangent = torch.tensor(
        [[0.2, -0.1, 0.4, 0.15, -0.25, 0.35]], dtype=torch.float64, requires_grad=True
    )
    pose = se3_exp_xyzw(tangent)
    loss = pose[..., :3].square().sum() + se3_log_xyzw(pose).square().sum()
    loss.backward()

    assert tangent.grad is not None
    assert torch.isfinite(tangent.grad).all()
    assert tangent.grad.abs().sum() > 0.0
