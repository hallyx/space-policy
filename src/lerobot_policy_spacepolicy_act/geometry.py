"""Differentiable XYZW quaternion and SE(3) primitives.

The canonical base-state contract represents a pose as
``[x, y, z, qx, qy, qz, qw]``.  The quaternion is XYZW and denotes
``T_world_from_base``: it rotates a vector expressed in the robot-base frame
into the world frame.  Measured canonical base twists remain world-frame
values and are deliberately not converted by this module.

For rollout deltas, :func:`relative_pose_body_xyzw` uses a right/body update:

``world_from_base_next = world_from_base @ base_from_base_next``.

Consequently the translational part of a relative pose (or an ``se3_exp``
delta) is expressed in the *current base frame*.  This is the convention used
by the reaction-model rollout; it prevents accidentally mixing the canonical
world-frame observation contract with a body-frame predicted increment.
"""

from __future__ import annotations

import math
from typing import Any

import torch


QUATERNION_XYZW_DIM = 4
ROTATION_VECTOR_DIM = 3
POSE_XYZW_DIM = 7
SE3_VECTOR_DIM = 6


def _floating_tensor(value: Any, name: str) -> torch.Tensor:
    """Return ``value`` as a floating tensor without detaching tensor inputs."""

    if isinstance(value, torch.Tensor):
        result = value
    else:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if not result.is_floating_point():
        result = result.to(dtype=torch.get_default_dtype())
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector_tensor(value: Any, width: int, name: str) -> torch.Tensor:
    result = _floating_tensor(value, name)
    if result.ndim < 1 or result.shape[-1] != width:
        raise ValueError(f"{name} must have shape (..., {width}), got {tuple(result.shape)}")
    return result


def _positive_norm(value: torch.Tensor, *, name: str, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if not bool(torch.all(norm > eps)):
        raise ValueError(f"{name} has a norm at or below {eps}")
    return norm


def _broadcast_pose_pair(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    return (
        left.expand((*shape, POSE_XYZW_DIM)),
        right.expand((*shape, POSE_XYZW_DIM)),
    )


def quaternion_normalize_xyzw(
    quaternion: Any,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Normalize an XYZW quaternion, preserving all leading batch dimensions."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    quat = _vector_tensor(quaternion, QUATERNION_XYZW_DIM, "quaternion")
    return quat / _positive_norm(quat, name="quaternion", eps=eps)


def quaternion_conjugate_xyzw(quaternion: Any) -> torch.Tensor:
    """Return the conjugate of an XYZW quaternion without normalizing it."""

    quat = _vector_tensor(quaternion, QUATERNION_XYZW_DIM, "quaternion")
    return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)


def quaternion_inverse_xyzw(
    quaternion: Any,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the inverse of a non-zero XYZW quaternion."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    quat = _vector_tensor(quaternion, QUATERNION_XYZW_DIM, "quaternion")
    squared_norm = (quat * quat).sum(dim=-1, keepdim=True)
    if not bool(torch.all(squared_norm > eps * eps)):
        raise ValueError(f"quaternion has a norm at or below {eps}")
    return quaternion_conjugate_xyzw(quat) / squared_norm


def quaternion_multiply_xyzw(
    left: Any,
    right: Any,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Compose XYZW quaternions using ``R(left) @ R(right)``.

    Leading dimensions are broadcast.  With unit inputs, this is the rotation
    composition appropriate for ``T_parent_from_mid @ T_mid_from_child``.
    """

    left_quat = _vector_tensor(left, QUATERNION_XYZW_DIM, "left quaternion")
    right_quat = _vector_tensor(right, QUATERNION_XYZW_DIM, "right quaternion")
    left_quat, right_quat = torch.broadcast_tensors(left_quat, right_quat)
    lx, ly, lz, lw = left_quat.unbind(dim=-1)
    rx, ry, rz, rw = right_quat.unbind(dim=-1)
    product = torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )
    return quaternion_normalize_xyzw(product) if normalize else product


def quaternion_rotate_vector_xyzw(quaternion: Any, vector: Any) -> torch.Tensor:
    """Rotate a 3-vector with an XYZW quaternion.

    A canonical ``world_from_base`` quaternion therefore maps a base-frame
    vector into world coordinates.
    """

    quat = quaternion_normalize_xyzw(quaternion)
    vector_tensor = _vector_tensor(vector, ROTATION_VECTOR_DIM, "vector")
    shape = torch.broadcast_shapes(quat.shape[:-1], vector_tensor.shape[:-1])
    quat = quat.expand((*shape, QUATERNION_XYZW_DIM))
    vector_tensor = vector_tensor.expand((*shape, ROTATION_VECTOR_DIM))
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    cross = torch.linalg.cross(xyz, vector_tensor, dim=-1)
    return vector_tensor + 2.0 * (w * cross + torch.linalg.cross(xyz, cross, dim=-1))


def quaternion_to_matrix_xyzw(quaternion: Any) -> torch.Tensor:
    """Convert XYZW quaternions to proper ``(..., 3, 3)`` rotation matrices."""

    x, y, z, w = quaternion_normalize_xyzw(quaternion).unbind(dim=-1)
    one = torch.ones_like(x)
    row0 = torch.stack(
        (one - 2.0 * (y.square() + z.square()), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        dim=-1,
    )
    row1 = torch.stack(
        (2.0 * (x * y + z * w), one - 2.0 * (x.square() + z.square()), 2.0 * (y * z - x * w)),
        dim=-1,
    )
    row2 = torch.stack(
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), one - 2.0 * (x.square() + y.square())),
        dim=-1,
    )
    return torch.stack((row0, row1, row2), dim=-2)


def so3_exp_xyzw(rotation_vector: Any) -> torch.Tensor:
    """Map an axis-angle vector in radians to an XYZW quaternion.

    ``torch.sinc`` supplies the analytic small-angle limit, so this operation
    is differentiable at the identity rotation.
    """

    omega = _vector_tensor(rotation_vector, ROTATION_VECTOR_DIM, "rotation_vector")
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    xyz = 0.5 * torch.sinc(angle / (2.0 * math.pi)) * omega
    w = torch.cos(0.5 * angle)
    return torch.cat((xyz, w), dim=-1)


def so3_log_xyzw(quaternion: Any, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Map an XYZW quaternion to its shortest axis-angle vector in radians."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    quat = quaternion_normalize_xyzw(quaternion, eps=eps)
    # q and -q describe the same rotation.  Choosing w >= 0 gives [0, pi].
    sign = torch.where(quat[..., 3:4] < 0.0, -torch.ones_like(quat[..., 3:4]), torch.ones_like(quat[..., 3:4]))
    quat = quat * sign
    xyz = quat[..., :3]
    sin_half_angle = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, quat[..., 3:4])
    scale = angle / sin_half_angle.clamp_min(eps)
    scale = torch.where(sin_half_angle > eps, scale, torch.full_like(scale, 2.0))
    return scale * xyz


def quaternion_relative_xyzw(start: Any, end: Any) -> torch.Tensor:
    """Return the right/body relative rotation ``R_start.T @ R_end``."""

    start_quat = quaternion_normalize_xyzw(start)
    end_quat = quaternion_normalize_xyzw(end)
    return quaternion_multiply_xyzw(
        quaternion_inverse_xyzw(start_quat),
        end_quat,
        normalize=True,
    )


def quaternion_geodesic_distance_xyzw(left: Any, right: Any) -> torch.Tensor:
    """Return the sign-invariant shortest SO(3) distance in radians, in ``[0, pi]``."""

    relative = quaternion_relative_xyzw(left, right)
    sin_half_angle = torch.linalg.vector_norm(relative[..., :3], dim=-1)
    cos_half_angle = relative[..., 3].abs()
    return 2.0 * torch.atan2(sin_half_angle, cos_half_angle)


def quaternion_slerp_xyzw(start: Any, end: Any, fraction: Any) -> torch.Tensor:
    """Geodesically interpolate between XYZW quaternions.

    ``fraction`` must broadcast with the quaternion batch shape.  A scalar
    fraction can therefore interpolate an arbitrary batch, while a vector can
    interpolate one pose at multiple fractions.
    """

    start_quat = quaternion_normalize_xyzw(start)
    end_quat = quaternion_normalize_xyzw(end)
    fraction_tensor = _floating_tensor(fraction, "fraction")
    shape = torch.broadcast_shapes(
        start_quat.shape[:-1], end_quat.shape[:-1], fraction_tensor.shape
    )
    start_quat = start_quat.expand((*shape, QUATERNION_XYZW_DIM))
    end_quat = end_quat.expand((*shape, QUATERNION_XYZW_DIM))
    fraction_tensor = fraction_tensor.expand(shape)
    relative = quaternion_relative_xyzw(start_quat, end_quat)
    delta = so3_exp_xyzw(so3_log_xyzw(relative) * fraction_tensor.unsqueeze(-1))
    return quaternion_multiply_xyzw(start_quat, delta, normalize=True)


def pose_compose_xyzw(left: Any, right: Any) -> torch.Tensor:
    """Compose poses as ``T_parent_from_mid @ T_mid_from_child``.

    Both inputs and the output use ``[position(3), quaternion_xyzw(4)]``.
    """

    left_pose = _vector_tensor(left, POSE_XYZW_DIM, "left pose")
    right_pose = _vector_tensor(right, POSE_XYZW_DIM, "right pose")
    left_pose, right_pose = _broadcast_pose_pair(left_pose, right_pose)
    position = left_pose[..., :3] + quaternion_rotate_vector_xyzw(
        left_pose[..., 3:], right_pose[..., :3]
    )
    quaternion = quaternion_multiply_xyzw(
        left_pose[..., 3:], right_pose[..., 3:], normalize=True
    )
    return torch.cat((position, quaternion), dim=-1)


def pose_inverse_xyzw(pose: Any) -> torch.Tensor:
    """Invert a pose represented as ``[position, quaternion_xyzw]``."""

    pose_tensor = _vector_tensor(pose, POSE_XYZW_DIM, "pose")
    inverse_quaternion = quaternion_inverse_xyzw(pose_tensor[..., 3:])
    inverse_position = -quaternion_rotate_vector_xyzw(
        inverse_quaternion, pose_tensor[..., :3]
    )
    return torch.cat((inverse_position, quaternion_normalize_xyzw(inverse_quaternion)), dim=-1)


def relative_pose_body_xyzw(start_world_from_base: Any, end_world_from_base: Any) -> torch.Tensor:
    """Return the body/right relative pose from two canonical world poses.

    The output is ``T_base_at_start_from_base_at_end`` and obeys
    ``end == pose_compose_xyzw(start, relative)``.
    """

    start = _vector_tensor(start_world_from_base, POSE_XYZW_DIM, "start_world_from_base")
    end = _vector_tensor(end_world_from_base, POSE_XYZW_DIM, "end_world_from_base")
    return pose_compose_xyzw(pose_inverse_xyzw(start), end)


def apply_body_delta_xyzw(world_from_base: Any, body_delta: Any) -> torch.Tensor:
    """Apply a body/right delta to a canonical ``T_world_from_base`` pose."""

    pose = _vector_tensor(world_from_base, POSE_XYZW_DIM, "world_from_base")
    delta = _vector_tensor(body_delta, POSE_XYZW_DIM, "body_delta")
    return pose_compose_xyzw(pose, delta)


def se3_exp_xyzw(tangent: Any) -> torch.Tensor:
    """Map a ``[..., 6]`` SE(3) tangent ``[rho, omega]`` to an XYZW pose.

    ``rho`` and ``omega`` use the same local frame.  This is the proper SE(3)
    exponential (including the SO(3) left Jacobian), rather than simply
    concatenating a translation and an axis-angle rotation.
    """

    tangent_tensor = _vector_tensor(tangent, SE3_VECTOR_DIM, "se3 tangent")
    rho = tangent_tensor[..., :3]
    omega = tangent_tensor[..., 3:]
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    angle_squared = angle.square()
    # (1 - cos(theta)) / theta^2, written with sinc for a stable theta -> 0 limit.
    coefficient_a = 0.5 * torch.sinc(angle / (2.0 * math.pi)).square()
    # (theta - sin(theta)) / theta^3.  The series avoids cancellation near zero.
    safe_squared = angle_squared.clamp_min(torch.finfo(angle.dtype).eps)
    regular_b = (1.0 - torch.sinc(angle / math.pi)) / safe_squared
    series_b = (
        1.0 / 6.0
        - angle_squared / 120.0
        + angle_squared.square() / 5040.0
        - angle_squared.pow(3) / 362880.0
    )
    coefficient_b = torch.where(angle < 1.0e-3, series_b, regular_b)
    omega_cross_rho = torch.linalg.cross(omega, rho, dim=-1)
    position = rho + coefficient_a * omega_cross_rho + coefficient_b * torch.linalg.cross(
        omega, omega_cross_rho, dim=-1
    )
    return torch.cat((position, so3_exp_xyzw(omega)), dim=-1)


def se3_log_xyzw(pose: Any) -> torch.Tensor:
    """Map an XYZW pose to its shortest ``[..., 6]`` tangent ``[rho, omega]``."""

    pose_tensor = _vector_tensor(pose, POSE_XYZW_DIM, "pose")
    position = pose_tensor[..., :3]
    omega = so3_log_xyzw(pose_tensor[..., 3:])
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    angle_squared = angle.square()
    half_angle = 0.5 * angle
    safe_sine = torch.sin(half_angle).clamp_min(torch.finfo(angle.dtype).eps)
    cot_half_angle = torch.cos(half_angle) / safe_sine
    safe_squared = angle_squared.clamp_min(torch.finfo(angle.dtype).eps)
    regular_c = (1.0 - half_angle * cot_half_angle) / safe_squared
    series_c = (
        1.0 / 12.0
        + angle_squared / 720.0
        + angle_squared.square() / 30240.0
        + angle_squared.pow(3) / 1209600.0
    )
    coefficient_c = torch.where(angle < 1.0e-3, series_c, regular_c)
    omega_cross_position = torch.linalg.cross(omega, position, dim=-1)
    rho = (
        position
        - 0.5 * omega_cross_position
        + coefficient_c * torch.linalg.cross(omega, omega_cross_position, dim=-1)
    )
    return torch.cat((rho, omega), dim=-1)


def pose_geodesic_interpolate_xyzw(start: Any, end: Any, fraction: Any) -> torch.Tensor:
    """Interpolate canonical poses along the body/right SE(3) geodesic."""

    start_pose = _vector_tensor(start, POSE_XYZW_DIM, "start pose")
    end_pose = _vector_tensor(end, POSE_XYZW_DIM, "end pose")
    fraction_tensor = _floating_tensor(fraction, "fraction")
    shape = torch.broadcast_shapes(
        start_pose.shape[:-1], end_pose.shape[:-1], fraction_tensor.shape
    )
    start_pose = start_pose.expand((*shape, POSE_XYZW_DIM))
    end_pose = end_pose.expand((*shape, POSE_XYZW_DIM))
    fraction_tensor = fraction_tensor.expand(shape)
    delta_tangent = se3_log_xyzw(relative_pose_body_xyzw(start_pose, end_pose))
    return apply_body_delta_xyzw(
        start_pose,
        se3_exp_xyzw(delta_tangent * fraction_tensor.unsqueeze(-1)),
    )


# Short aliases keep call sites readable while the ``_xyzw`` names make the
# storage order explicit at module boundaries.
quaternion_normalize = quaternion_normalize_xyzw
quaternion_multiply = quaternion_multiply_xyzw
quaternion_geodesic_distance = quaternion_geodesic_distance_xyzw
pose_compose = pose_compose_xyzw
pose_inverse = pose_inverse_xyzw
relative_pose_body = relative_pose_body_xyzw
apply_body_delta = apply_body_delta_xyzw


__all__ = [
    "POSE_XYZW_DIM",
    "QUATERNION_XYZW_DIM",
    "ROTATION_VECTOR_DIM",
    "SE3_VECTOR_DIM",
    "apply_body_delta",
    "apply_body_delta_xyzw",
    "pose_compose",
    "pose_compose_xyzw",
    "pose_geodesic_interpolate_xyzw",
    "pose_inverse",
    "pose_inverse_xyzw",
    "quaternion_conjugate_xyzw",
    "quaternion_geodesic_distance",
    "quaternion_geodesic_distance_xyzw",
    "quaternion_inverse_xyzw",
    "quaternion_multiply",
    "quaternion_multiply_xyzw",
    "quaternion_normalize",
    "quaternion_normalize_xyzw",
    "quaternion_relative_xyzw",
    "quaternion_rotate_vector_xyzw",
    "quaternion_slerp_xyzw",
    "quaternion_to_matrix_xyzw",
    "relative_pose_body",
    "relative_pose_body_xyzw",
    "se3_exp_xyzw",
    "se3_log_xyzw",
    "so3_exp_xyzw",
    "so3_log_xyzw",
]
