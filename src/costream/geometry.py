"""Rigid transforms act on column vectors; translations use metres."""

import numpy as np
from scipy.spatial.transform import Rotation


def transform(value):
    """Return an owned, validated SE(3) matrix (never silently repair it)."""
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('transform must be a finite 4x4 matrix')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
        raise ValueError('transform must have a proper rotation and homogeneous last row')
    return matrix


def compose(anchor, trajectory, tactile):
    """Paper equations: W_T_cmd = W_T_I @ I_T_traj @ DeltaT_tactile.

    The right factor is a correction in the nominal end-effector body frame.
    A residual from another frame must be converted before calling this function.
    """
    return transform(anchor) @ transform(trajectory) @ transform(tactile)


def world_delta_to_body(nominal, delta):
    """Adapt ReKep's [world translation, world rotvec] to a right factor.

    ReKep adds translation in world coordinates and left-multiplies orientation.
    This conversion preserves that behavior, including for nonzero translations
    and noncommuting rotations. It does not reinterpret a world delta as a body
    delta or rotate the nominal position about the world origin.
    """
    nominal = transform(nominal)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (6,) or not np.isfinite(delta).all():
        raise ValueError('world delta must be six finite values (metres, radians)')
    rotation = nominal[:3, :3]
    out = np.eye(4)
    out[:3, 3] = rotation.T @ delta[:3]
    out[:3, :3] = rotation.T @ Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
    return out
