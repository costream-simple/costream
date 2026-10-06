"""Deterministic synthetic inputs shared by the pipeline demo and the tests.

Nothing here models a real camera, gel, or rollout; it only produces inputs
whose ground-truth geometry is known exactly.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .predictive.tracks import TrackSet, project
from .reactive.maps import TactileMaps


def pose(xyz=(0, 0, 0), rotvec=(0, 0, 0)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = xyz
    return out


def camera_pose():
    """World <- camera 0: 0.8 m above the table, looking down, slightly yawed."""
    out = np.eye(4)
    out[:3, :3] = (Rotation.from_euler('x', np.pi) * Rotation.from_euler('z', .3)).as_matrix()
    out[:3, 3] = (.45, 0., .8)
    return out


def camera_intrinsics(shape=(240, 320), focal=300.):
    return np.array([[focal, 0., shape[1] / 2], [0., focal, shape[0] / 2], [0., 0., 1.]])


def box_points(size=(.04, .03, .02)):
    """Eight box corners centred on the object origin (centrally symmetric)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return signs * np.asarray(size) / 2


def rigid_tracks(object_points, W_T_O, *, W_T_C0, scale=1., shape=(240, 320), background_depth=.8):
    """Tracks of points rigidly attached to an object with world poses W_T_O[t].

    Returns (TrackSet in reconstruction units = metric / scale, measured metric
    frame-0 depth). Both depth maps are a flat plane, which pins the scale.
    """
    homogeneous = np.c_[np.asarray(object_points, dtype=float), np.ones(len(object_points))]
    C0_T_W = np.linalg.inv(W_T_C0)
    metric = np.stack([(C0_T_W @ np.asarray(T) @ homogeneous.T).T[:, :3] for T in W_T_O])
    measured = np.full(shape, float(background_depth))
    tracks = TrackSet(metric / scale, np.ones(metric.shape[:2], bool), measured / scale,
                      project(metric[0], camera_intrinsics(shape)))
    return tracks, measured


MM_PER_PIXEL = .0634
TACTILE_SHAPE = (180, 240)


def tactile_maps(*, dx_mm=0., dy_mm=0., yaw=0., pressed=True, shape=TACTILE_SHAPE, mm_per_pixel=MM_PER_PIXEL):
    """Two Gaussian bumps (asymmetric, so yaw is observable) moved rigidly in the gel plane.

    The contact shifts by (dx_mm, dy_mm) and rotates by ``yaw`` about the sensor
    z axis at the image centre, so registration against the unmoved frame must
    return S1_T_S0 with exactly that translation and yaw.
    """
    rows, cols = np.mgrid[:shape[0], :shape[1]].astype(float)
    x = (cols - shape[1] / 2 + .5) * mm_per_pixel
    y = (rows - shape[0] / 2 + .5) * mm_per_pixel
    c, s = np.cos(yaw), np.sin(yaw)
    x_ref = c * (x - dx_mm) + s * (y - dy_mm)
    y_ref = -s * (x - dx_mm) + c * (y - dy_mm)
    height = (30 * np.exp(-((x_ref - .6) ** 2 + y_ref ** 2) / (2 * 1.2 ** 2))
              + 18 * np.exp(-((x_ref + 1.4) ** 2 + (y_ref - .8) ** 2) / (2 * .8 ** 2)))
    if not pressed:
        height = np.zeros(shape)
    grad_y, grad_x = np.gradient(height)
    normal = np.dstack([-grad_x, -grad_y, np.ones(shape)])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return TactileMaps(normal, height, height > 4.)


class PassthroughReconstructor:
    """A TactileReconstructor whose 'images' are already TactileMaps."""

    def reconstruct(self, image):
        return image
