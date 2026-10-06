"""Point tracks lifted from a generated rollout, expressed in the frame-0 camera."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

_SPATRACKER_KEYS = ('coords', 'visibs', 'depths', 'intrinsics', 'extrinsics')


@dataclass(frozen=True)
class TrackSet:
    """3D tracks of one rollout.

    points: [T,N,3] in the frame-0 camera frame, reconstruction units (scale unknown).
    visible: [T,N] bool; invisible points may be NaN.
    depth0: [H,W] predicted frame-0 depth, same units as points.
    uv0: optional [N,2] frame-0 pixel (column, row); NaN where unknown.
    """

    points: np.ndarray
    visible: np.ndarray
    depth0: np.ndarray
    uv0: Optional[np.ndarray] = None

    def __post_init__(self):
        points = np.array(self.points, dtype=float)
        if points.ndim != 3 or points.shape[2] != 3 or not points.shape[0] or not points.shape[1]:
            raise ValueError('points must have shape [T,N,3] with T, N > 0')
        visible = np.array(self.visible)
        if visible.shape != points.shape[:2]:
            raise ValueError('visible must have shape [T,N]')
        if visible.dtype != bool:
            raise ValueError('visible must be boolean; threshold visibility probabilities first')
        if not np.isfinite(points[visible]).all():
            raise ValueError('visible track points must be finite')
        depth0 = np.array(self.depth0, dtype=float)
        if depth0.ndim != 2:
            raise ValueError('depth0 must be a 2D depth map')
        uv0 = None
        if self.uv0 is not None:
            uv0 = np.array(self.uv0, dtype=float)
            if uv0.shape != (points.shape[1], 2):
                raise ValueError('uv0 must have shape [N,2]')
        for name, value in (('points', points), ('visible', visible), ('depth0', depth0), ('uv0', uv0)):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)


def project(points, K):
    """Pinhole projection of [N,3] camera points to [N,2] pixels; NaN at or behind the camera."""
    points = np.asarray(points, dtype=float)
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3):
        raise ValueError('intrinsics must be a 3x3 matrix')
    uv = np.full((len(points), 2), np.nan)
    front = points[:, 2] > 0
    pixels = points[front] @ K.T
    uv[front] = pixels[:, :2] / pixels[:, 2:]
    return uv


def load_spatracker_npz(path, *, visibility_threshold=.5):
    """Read SpaTrackerV2 ``inference.py`` output (``result.npz``) into camera-0 coordinates.

    Keys: coords [T,N,3] (reconstruction world), visibs [T,N] or [T,N,1] (bool or
    probability), depths [T,H,W], intrinsics [T,3,3], extrinsics [T,4,4]
    (world -> camera), optional tracks2d [T,N,>=2]. Only this file format is
    read; SpaTrackerV2 code (CC BY-NC) is never imported.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _SPATRACKER_KEYS if key not in data]
        if missing:
            raise KeyError(f'{path}: missing SpaTrackerV2 keys {missing}; expected {list(_SPATRACKER_KEYS)}')
        coords = np.asarray(data['coords'], dtype=float)
        visibs = np.asarray(data['visibs'])
        depths = np.asarray(data['depths'], dtype=float)
        intrinsics = np.asarray(data['intrinsics'], dtype=float)
        extrinsics = np.asarray(data['extrinsics'], dtype=float)
        tracks2d = np.asarray(data['tracks2d'], dtype=float) if 'tracks2d' in data else None
    if coords.ndim != 3 or extrinsics.shape != (len(coords), 4, 4):
        raise ValueError('coords must be [T,N,3] and extrinsics [T,4,4] with matching T')
    if depths.ndim != 3 or intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError('depths must be [T,H,W] and intrinsics [T,3,3]')
    if visibs.ndim == 3 and visibs.shape[-1] == 1:
        visibs = visibs[..., 0]
    visible = visibs if visibs.dtype == bool else visibs.astype(float) > visibility_threshold
    C0_T_recon = extrinsics[0]
    points = coords @ C0_T_recon[:3, :3].T + C0_T_recon[:3, 3]
    uv0 = tracks2d[0, :, :2] if tracks2d is not None else project(points[0], intrinsics[0])
    return TrackSet(points, visible, depths[0], uv0)
