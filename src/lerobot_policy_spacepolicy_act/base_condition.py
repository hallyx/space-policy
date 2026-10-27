"""机器人浮动基座条件的稳定数值投影。"""

from __future__ import annotations

from typing import Any

import numpy as np


BASE9_DIM = 9
BASE15_DIM = 15
BASE15_SCHEMA_VERSION = "robot_base15_world_xyzw_rot6d_cols_v1"
BASE15_ROTATION_CONVENTION = "concat(R_world_from_base[:,0], R_world_from_base[:,1])"
BASE15_VELOCITY_FRAME = "world"


def quaternion_xyzw_to_rotation_6d(quaternion: Any) -> np.ndarray:
    """将 XYZW 四元数转为连续 6D 旋转表示。

    输出严格定义为 ``concat(R[:, 0], R[:, 1])``，其中 ``R`` 是
    ``world_from_robot_base`` 旋转矩阵。也就是基座局部 x、y 轴在 world
    坐标系中的方向；四元数会先归一化，因此 ``q`` 与 ``-q`` 输出相同。
    """
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"base quaternion 必须是 XYZW shape=(4,)，实际为 {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("base quaternion 包含 NaN/Inf")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise ValueError("base quaternion 范数过小，无法构造 rotation_6d")
    x, y, z, w = quat / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rotation_6d = np.concatenate((rotation[:, 0], rotation[:, 1])).astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(rotation_6d)


def base_environment_state_from_pose_velocity(
    base_pose: Any,
    base_velocity: Any,
    *,
    output_dim: int,
) -> np.ndarray:
    """从采集字段生成 base9 或 base15 条件。

    ``base_pose`` 必须为 ``[x,y,z,qx,qy,qz,qw]``；``base_velocity`` 必须为
    world frame 下的 ``[vx,vy,vz,wx,wy,wz]``。
    """
    pose = np.asarray(base_pose, dtype=np.float32).reshape(-1)
    velocity = np.asarray(base_velocity, dtype=np.float32).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"diagnostics.base_pose 必须是 7D，实际为 {pose.shape}")
    if velocity.shape != (6,):
        raise ValueError(f"diagnostics.base_velocity 必须是 6D，实际为 {velocity.shape}")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
        raise ValueError("base pose/velocity 包含 NaN/Inf")

    if output_dim == BASE9_DIM:
        values = np.concatenate((pose[:3], velocity))
    elif output_dim == BASE15_DIM:
        rotation_6d = quaternion_xyzw_to_rotation_6d(pose[3:7])
        values = np.concatenate((pose[:3], rotation_6d, velocity))
    else:
        raise ValueError(f"仅支持 base9/base15，实际 output_dim={output_dim}")
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (output_dim,):
        raise AssertionError(f"base condition shape 异常：{result.shape}")
    return np.ascontiguousarray(result)
