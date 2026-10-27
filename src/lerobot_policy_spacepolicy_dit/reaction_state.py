"""Explicit, read-only state contracts for reaction-aware ACT experiments.

This module deliberately has no dependency on an ACT policy, processor, or training
entrypoint.  It names the source-frame semantics needed by the reaction model before
any model is allowed to consume the data:

* base pose is ``world_from_base`` as ``[x, y, z, qx, qy, qz, qw]``;
* base twist is world-frame ``[vx, vy, vz, wx, wy, wz]``;
* ``q``/``dq`` are the *measured* full 26-DoF articulation values; and
* Left15 velocity is a desired drive velocity, never a measured velocity.

Arrays are copied and made non-writeable at the boundary.  That makes a transition
builder safe to use as a read-only dataset derivation rather than another mutable
view of canonical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot_policy_spacepolicy_dit.base_condition import (
    BASE9_DIM,
    BASE15_DIM,
    base_environment_state_from_pose_velocity,
)


LEFT15_DIM = 15
FULL_ROBOT_DOF = 26
BASE_POSE_DIM = 7
BASE_TWIST_DIM = 6
TASK_POSE_DIM = 7
EE_POSE_PAIR_DIM = 14
REACTION_MODEL_STATE_DIM = BASE_POSE_DIM + BASE_TWIST_DIM + 2 * FULL_ROBOT_DOF + LEFT15_DIM


def _readonly_float_vector(value: Any, *, name: str, width: int) -> np.ndarray:
    """Validate, copy, and freeze one canonical numeric vector."""

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape=({width},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_float_matrix(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Validate, copy, and freeze one canonical numeric matrix."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape={shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_scalar(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return result


def normalize_quaternion_xyzw(value: Any, *, name: str = "quaternion") -> np.ndarray:
    """Return a canonical unit XYZW quaternion without changing its represented pose."""

    quaternion = _readonly_float_vector(value, name=name, width=4).astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError(f"{name} has near-zero norm")
    result = (quaternion / norm).astype(np.float32)
    # q and -q denote the same rotation.  One deterministic sign avoids a false
    # discontinuity in a relative-pose target or audit.
    if result[3] < 0.0:
        result *= -1.0
    result.setflags(write=False)
    return result


def rotation_matrix_world_from_base(base_pose_world_xyzw: Any) -> np.ndarray:
    """Project a canonical base pose to ``R_world_from_base``.

    The output is a detached, read-only ``float32[3, 3]`` matrix.  This is a
    diagnostic/projection helper; reaction-model SO(3) Exp/Log remains in the
    dedicated differentiable math module.
    """

    pose = _readonly_float_vector(
        base_pose_world_xyzw,
        name="base_pose_world_xyzw",
        width=BASE_POSE_DIM,
    )
    x, y, z, w = normalize_quaternion_xyzw(pose[3:], name="base_pose_world_xyzw[3:]")
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    rotation.setflags(write=False)
    return rotation


def quaternion_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    """Hamilton product for unit XYZW quaternions, with a deterministic sign."""

    lx, ly, lz, lw = normalize_quaternion_xyzw(left, name="left_quaternion")
    rx, ry, rz, rw = normalize_quaternion_xyzw(right, name="right_quaternion")
    product = np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float32,
    )
    return normalize_quaternion_xyzw(product, name="quaternion_product")


def quaternion_conjugate_xyzw(value: Any) -> np.ndarray:
    """Conjugate a unit XYZW quaternion."""

    x, y, z, w = normalize_quaternion_xyzw(value)
    result = np.array([-x, -y, -z, w], dtype=np.float32)
    result.setflags(write=False)
    return result


def relative_base_pose_body_xyzw(
    current_base_pose_world_xyzw: Any,
    future_base_pose_world_xyzw: Any,
) -> np.ndarray:
    """Return future-base displacement in the current base frame.

    The result is ``[translation_body(3), q_current_from_future_xyzw(4)]``.  It is
    intentionally quaternion-valued here; the trainable SO(3) Log mapping belongs to
    the differentiable math primitive rather than this NumPy source-contract module.
    """

    current = _readonly_float_vector(
        current_base_pose_world_xyzw,
        name="current_base_pose_world_xyzw",
        width=BASE_POSE_DIM,
    )
    future = _readonly_float_vector(
        future_base_pose_world_xyzw,
        name="future_base_pose_world_xyzw",
        width=BASE_POSE_DIM,
    )
    rotation = rotation_matrix_world_from_base(current)
    translation_body = rotation.T @ (future[:3] - current[:3])
    relative_quaternion = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(current[3:]),
        future[3:],
    )
    result = np.concatenate((translation_body, relative_quaternion)).astype(np.float32)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Unambiguous source identity for one canonical pre-action frame."""

    episode_index: int
    frame_index: int
    global_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_index",
            _nonnegative_int(self.episode_index, name="episode_index"),
        )
        object.__setattr__(
            self,
            "frame_index",
            _nonnegative_int(self.frame_index, name="frame_index"),
        )
        object.__setattr__(
            self,
            "global_index",
            _nonnegative_int(self.global_index, name="global_index"),
        )


@dataclass(frozen=True, slots=True)
class Left15Command:
    """One same-frame, post-observation Left15 servo command.

    ``arm_desired_velocity`` is the controller drive target from the canonical data;
    it must not be substituted with ``RobotState.joint_velocity``.
    """

    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _readonly_float_vector(self.values, name="left15_command", width=LEFT15_DIM),
        )

    @property
    def arm_position_target(self) -> np.ndarray:
        return self.values[:7]

    @property
    def arm_desired_velocity(self) -> np.ndarray:
        return self.values[7:14]

    @property
    def gripper_position_target(self) -> np.ndarray:
        return self.values[14:15]


@dataclass(frozen=True, slots=True)
class RobotState:
    """Full physically measured robot/base state at one pre-action frame."""

    base_pose_world_xyzw: np.ndarray
    base_twist_world: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray

    def __post_init__(self) -> None:
        pose = _readonly_float_vector(
            self.base_pose_world_xyzw,
            name="base_pose_world_xyzw",
            width=BASE_POSE_DIM,
        )
        # Validate against an actually representable orientation, while retaining the
        # source quaternion values byte-for-byte (aside from safe array ownership).
        normalize_quaternion_xyzw(pose[3:], name="base_pose_world_xyzw[3:]")
        object.__setattr__(self, "base_pose_world_xyzw", pose)
        object.__setattr__(
            self,
            "base_twist_world",
            _readonly_float_vector(
                self.base_twist_world,
                name="base_twist_world",
                width=BASE_TWIST_DIM,
            ),
        )
        object.__setattr__(
            self,
            "joint_position",
            _readonly_float_vector(
                self.joint_position,
                name="joint_position",
                width=FULL_ROBOT_DOF,
            ),
        )
        object.__setattr__(
            self,
            "joint_velocity",
            _readonly_float_vector(
                self.joint_velocity,
                name="joint_velocity",
                width=FULL_ROBOT_DOF,
            ),
        )

    @classmethod
    def from_canonical_diagnostics(
        cls,
        *,
        base_pose: Any,
        base_velocity: Any,
        raw_joint_state: Any,
        measured_joint_velocity: Any,
    ) -> "RobotState":
        """Project the four canonical diagnostic columns without reinterpretation."""

        return cls(
            base_pose_world_xyzw=base_pose,
            base_twist_world=base_velocity,
            joint_position=raw_joint_state,
            joint_velocity=measured_joint_velocity,
        )

    @property
    def base_twist_body(self) -> np.ndarray:
        """World-frame measured twist expressed in the current base frame."""

        rotation = rotation_matrix_world_from_base(self.base_pose_world_xyzw)
        result = np.concatenate(
            (
                rotation.T @ self.base_twist_world[:3],
                rotation.T @ self.base_twist_world[3:],
            )
        ).astype(np.float32)
        result.setflags(write=False)
        return result

    def project_base_condition(self, *, output_dim: int) -> np.ndarray:
        """Use the identical base9/base15 online projection for offline audits."""

        if output_dim not in {BASE9_DIM, BASE15_DIM}:
            raise ValueError(f"output_dim must be {BASE9_DIM} or {BASE15_DIM}, got {output_dim}")
        result = base_environment_state_from_pose_velocity(
            self.base_pose_world_xyzw,
            self.base_twist_world,
            output_dim=output_dim,
        )
        result.setflags(write=False)
        return result

    def reaction_features(self) -> dict[str, np.ndarray]:
        """Return exactly the physical state inputs allowed to a v1 reaction model."""

        return {
            "base_pose_world_xyzw": self.base_pose_world_xyzw,
            "base_twist_world": self.base_twist_world,
            "q": self.joint_position,
            "dq": self.joint_velocity,
        }


def pack_reaction_model_state(
    robot_state: RobotState,
    previous_left15_command: Left15Command,
) -> np.ndarray:
    """Pack the explicit v1 reaction state in one stable 80D field order.

    ``[base_pose7, base_twist6, q26, dq26, previous_left15_command15]`` is kept
    here rather than inferred by a caller, so offline data and the separate reaction
    model cannot silently disagree about a slice boundary.
    """

    if not isinstance(robot_state, RobotState):
        raise TypeError("robot_state must be a RobotState")
    if not isinstance(previous_left15_command, Left15Command):
        raise TypeError("previous_left15_command must be a Left15Command")
    result = np.concatenate(
        (
            robot_state.base_pose_world_xyzw,
            robot_state.base_twist_world,
            robot_state.joint_position,
            robot_state.joint_velocity,
            previous_left15_command.values,
        )
    ).astype(np.float32)
    if result.shape != (REACTION_MODEL_STATE_DIM,):
        raise AssertionError(f"packed reaction state has unexpected shape {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TaskState:
    """Current-frame task diagnostics retained for audit only.

    These fields are intentionally separate from ``RobotState``.  V1 reaction-model
    features must not consume target/EE diagnostics as privileged policy inputs.
    """

    relative_target_pose: np.ndarray
    delta_ee_target: np.ndarray
    left_ee_pose_world_xyzw: np.ndarray
    right_ee_pose_world_xyzw: np.ndarray
    grasp_success: float

    def __post_init__(self) -> None:
        relative_target = _readonly_float_vector(
            self.relative_target_pose,
            name="relative_target_pose",
            width=TASK_POSE_DIM,
        )
        delta = _readonly_float_vector(
            self.delta_ee_target,
            name="delta_ee_target",
            width=TASK_POSE_DIM,
        )
        left = _readonly_float_vector(
            self.left_ee_pose_world_xyzw,
            name="left_ee_pose_world_xyzw",
            width=TASK_POSE_DIM,
        )
        right = _readonly_float_vector(
            self.right_ee_pose_world_xyzw,
            name="right_ee_pose_world_xyzw",
            width=TASK_POSE_DIM,
        )
        # The source values are preserved, but malformed zero quaternions are rejected
        # before a diagnostic is allowed into an audit record.
        for name, pose in (
            ("relative_target_pose", relative_target),
            ("delta_ee_target", delta),
            ("left_ee_pose_world_xyzw", left),
            ("right_ee_pose_world_xyzw", right),
        ):
            normalize_quaternion_xyzw(pose[3:], name=f"{name}[3:]")
        object.__setattr__(self, "relative_target_pose", relative_target)
        object.__setattr__(self, "delta_ee_target", delta)
        object.__setattr__(self, "left_ee_pose_world_xyzw", left)
        object.__setattr__(self, "right_ee_pose_world_xyzw", right)
        object.__setattr__(
            self,
            "grasp_success",
            _finite_scalar(self.grasp_success, name="grasp_success"),
        )

    @classmethod
    def from_canonical_diagnostics(
        cls,
        *,
        relative_target_pose: Any,
        delta_ee_target: Any,
        ee_pose: Any,
        grasp_success: Any,
    ) -> "TaskState":
        """Split canonical ``diagnostics.ee_pose[14]`` into left/right poses."""

        pair = _readonly_float_vector(
            ee_pose,
            name="ee_pose",
            width=EE_POSE_PAIR_DIM,
        )
        return cls(
            relative_target_pose=relative_target_pose,
            delta_ee_target=delta_ee_target,
            left_ee_pose_world_xyzw=pair[:7],
            right_ee_pose_world_xyzw=pair[7:14],
            grasp_success=grasp_success,
        )


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    """One completely specified canonical 50 Hz pre-action source frame."""

    identity: FrameIdentity
    timestamp_s: float
    monotonic_time_s: float
    simulation_step: int
    control_state: int
    robot: RobotState
    action: Left15Command
    task_diagnostics: TaskState

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite_scalar(self.timestamp_s, name="timestamp_s"))
        object.__setattr__(
            self,
            "monotonic_time_s",
            _finite_scalar(self.monotonic_time_s, name="monotonic_time_s"),
        )
        object.__setattr__(
            self,
            "simulation_step",
            _nonnegative_int(self.simulation_step, name="simulation_step"),
        )
        if isinstance(self.control_state, bool):
            raise ValueError("control_state must be an integer, not bool")
        object.__setattr__(self, "control_state", int(self.control_state))


__all__ = [
    "BASE_POSE_DIM",
    "BASE_TWIST_DIM",
    "EE_POSE_PAIR_DIM",
    "FULL_ROBOT_DOF",
    "LEFT15_DIM",
    "REACTION_MODEL_STATE_DIM",
    "TASK_POSE_DIM",
    "CanonicalFrame",
    "FrameIdentity",
    "Left15Command",
    "RobotState",
    "TaskState",
    "normalize_quaternion_xyzw",
    "pack_reaction_model_state",
    "quaternion_conjugate_xyzw",
    "quaternion_multiply_xyzw",
    "relative_base_pose_body_xyzw",
    "rotation_matrix_world_from_base",
]
