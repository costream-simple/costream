"""Lift selected point tracks to a task-frame trajectory I_T_traj (paper Sec. 3.3)."""

import numpy as np

from ..geometry import transform
from ..trajectory import Trajectory


def select_tracks(tracks, *, mask=None, min_visible_frac=.5):
    """Indices of tracks on the manipulated object.

    Tracks visible in fewer than ``min_visible_frac`` of frames are dropped. A
    frame-0 boolean ``mask`` keeps tracks visible in frame 0 whose rounded ``uv0``
    falls inside it; tracks without a finite in-image ``uv0`` are excluded, never guessed.
    """
    if not 0 <= min_visible_frac <= 1:
        raise ValueError('min_visible_frac must be in [0, 1]')
    keep = tracks.visible.mean(axis=0) >= min_visible_frac
    n_in_mask = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.dtype != bool:
            raise ValueError('mask must be a 2D boolean image')
        if mask.shape != tracks.depth0.shape:
            raise ValueError(f'mask shape {mask.shape} must match the tracker frame (depth0 shape '
                             f'{tracks.depth0.shape}); resample it to the tracked video first')
        if tracks.uv0 is None:
            raise ValueError('mask selection needs frame-0 pixel coordinates (TrackSet.uv0)')
        in_mask = np.zeros(len(keep), dtype=bool)
        finite = np.flatnonzero(np.isfinite(tracks.uv0).all(axis=1))
        cols, rows = np.rint(tracks.uv0[finite]).astype(int).T
        inside = (cols >= 0) & (cols < mask.shape[1]) & (rows >= 0) & (rows < mask.shape[0])
        in_mask[finite[inside]] = mask[rows[inside], cols[inside]]
        n_in_mask = int(in_mask.sum())
        keep &= in_mask & tracks.visible[0]
    idx = np.flatnonzero(keep)
    if not idx.size:
        raise ValueError('no tracks survived selection; check the mask and visibility threshold')
    return idx, {'n_tracks_total': int(len(keep)), 'n_tracks_in_mask': n_in_mask,
                 'n_tracks_selected': int(idx.size)}


def centroid_motion(tracks, idx):
    """Per-frame median of the visible selected points: [T,3], reconstruction units.

    The median, not the mean: points drifting onto the background are common in
    generated video.
    """
    out = np.empty((len(tracks.points), 3))
    for t, (points, visible) in enumerate(zip(tracks.points[:, idx], tracks.visible[:, idx])):
        if not visible.any():
            raise ValueError(f'no selected track is visible in frame {t}')
        out[t] = np.median(points[visible], axis=0)
    return out


def rigid_motion(tracks, idx, *, min_points=3):
    """Per-frame rigid motion M(t) with points_t ≈ M(t) @ points_0 (Kabsch).

    Uses selected points visible in both frame 0 and frame t. Returns
    ([T,4,4] in reconstruction units, diagnostics with per-frame RMS residual
    and point count).
    """
    if min_points < 3:
        raise ValueError('a rigid fit needs at least three points')
    motions, residuals, counts = [], [], []
    for t in range(len(tracks.points)):
        common = idx[tracks.visible[0, idx] & tracks.visible[t, idx]]
        source, target = tracks.points[0, common], tracks.points[t, common]
        spread = (np.linalg.svd(source - source.mean(axis=0), compute_uv=False)
                  if len(common) else np.zeros(1))
        if len(common) < min_points or spread.size < 2 or spread[1] <= 1e-6 * spread[0]:
            raise ValueError(f'frame {t}: too few or collinear common points for a rigid fit')
        motion = _kabsch(source, target)
        fitted = source @ motion[:3, :3].T + motion[:3, 3]
        motions.append(motion)
        residuals.append(float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1)))))
        counts.append(int(len(common)))
    return np.array(motions), {'rms_residual': residuals, 'n_points': counts}


def _kabsch(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    flip = 1. if np.linalg.det(vt.T @ u.T) >= 0 else -1.
    rotation = vt.T @ np.diag([1., 1., flip]) @ u.T
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = target_mean - rotation @ source_mean
    return out


def lift_to_task_frame(tracks, idx, *, scale, W_T_C0, W_T_I, times, rotation='fixed', R_task=None,
                       O_T_E=None):
    """Return (Trajectory of I_T_traj, diagnostics).

    rotation='fixed' (paper default): translation from the per-frame median; every
    rotation is R_task (identity if omitted). rotation='kabsch' (opt-in): the
    frame-0 pose is (median, R_task) and later poses follow the fitted rigid
    motion, I_T_obj(t) = I_T_C0 @ M(t) @ inv(I_T_C0) @ I_T_obj(0). The nominal
    end-effector is I_T_traj(t) = I_T_obj(t) @ O_T_E. ``times`` is a positive
    frame period in seconds or one time per frame; it is never inferred.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive and finite')
    I_T_C0 = np.linalg.inv(transform(W_T_I)) @ transform(W_T_C0)
    start = np.eye(4)
    if R_task is not None:
        start[:3, :3] = R_task
    start = transform(start)
    O_T_E = np.eye(4) if O_T_E is None else transform(O_T_E)
    centroids = centroid_motion(tracks, idx) * scale
    times = _frame_times(times, len(centroids))
    diagnostics = {'rotation_mode': rotation}
    if rotation == 'fixed':
        I_T_obj = np.tile(start, (len(centroids), 1, 1))
        I_T_obj[:, :3, 3] = centroids @ I_T_C0[:3, :3].T + I_T_C0[:3, 3]
    elif rotation == 'kabsch':
        motions, fit = rigid_motion(tracks, idx)
        motions[:, :3, 3] *= scale
        start[:3, 3] = I_T_C0[:3, :3] @ centroids[0] + I_T_C0[:3, 3]
        I_T_obj = I_T_C0 @ motions @ np.linalg.inv(I_T_C0) @ start
        diagnostics['rms_residual_m'] = [residual * scale for residual in fit['rms_residual']]
        diagnostics['n_points'] = fit['n_points']
    else:
        raise ValueError("rotation must be 'fixed' or 'kabsch'")
    return Trajectory(times, I_T_obj @ O_T_E), diagnostics


def _frame_times(times, count):
    values = np.asarray(times, dtype=float)
    if values.ndim == 0:
        if not np.isfinite(values) or values <= 0:
            raise ValueError('frame period must be positive and finite')
        return np.arange(count) * float(values)
    if values.shape != (count,):
        raise ValueError(f'need one time per frame ({count}), got shape {values.shape}')
    return values


def deviation_from_waypoints(trajectory, waypoints):
    """Closest approach (mm) of the trajectory to each task-frame waypoint [K,3].

    Closest approach rather than index-matched distance: a rollout has no timing
    correspondence with a taught schedule.
    """
    waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    if not len(waypoints) or not np.isfinite(waypoints).all():
        raise ValueError('waypoints must be a nonempty finite [K,3] array')
    positions = trajectory.poses[:, :3, 3]
    per_waypoint = []
    for k, waypoint in enumerate(waypoints):
        distances = np.linalg.norm(positions - waypoint, axis=1)
        j = int(np.argmin(distances))
        per_waypoint.append({'waypoint_index': k, 'closest_approach_mm': float(distances[j] * 1000),
                             'at_frame': j})
    values = np.array([entry['closest_approach_mm'] for entry in per_waypoint])
    return {'per_waypoint': per_waypoint, 'median_closest_approach_mm': float(np.median(values)),
            'max_closest_approach_mm': float(values.max())}
