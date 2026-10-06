"""NormalFlow tactile registration (Huang et al., RA-L 2024), ported to SciPy.

Adapted from https://github.com/rpl-cmu/normalflow (MIT licence, see NOTICE).
cv2.remap and cv2.Sobel are replaced by SciPy equivalents with the same border
handling; tests/test_normalflow_parity.py checks agreement with upstream.
"""

import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import correlate1d
from scipy.spatial.transform import Rotation

from ..geometry import transform

_SMOOTH = np.array([1., 4., 6., 4., 1.])
_DERIVATIVE = np.array([-1., -2., 0., 2., 1.])


class InsufficientOverlapError(RuntimeError):
    def __init__(self, message='insufficient shared contact between frames for NormalFlow registration'):
        super().__init__(message)


def register(ref, cur, *, mm_per_pixel, init=None, n_samples=5000, max_iters=50, rng=None):
    """Return S1_T_S0: maps reference contact points (sensor frame, metres) into the current frame.

    ``ref`` and ``cur`` are TactileMaps of one size. ``init`` is an initial
    guess (identity if omitted). ``n_samples`` reference contact pixels are drawn
    with ``rng`` (all pixels if None). In-plane motion and rotation come from
    Gauss-Newton on normals; z translation from the mean height difference.
    """
    mm = float(mm_per_pixel)
    if not np.isfinite(mm) or mm <= 0:
        raise ValueError('mm_per_pixel must be positive and finite')
    if ref.height.shape != cur.height.shape:
        raise ValueError('reference and current maps differ in size')
    T = np.eye(4) if init is None else transform(init)
    points = _pointcloud(ref.height, mm)[ref.contact.reshape(-1)]
    rows, cols = np.nonzero(ref.contact)
    if len(points) < 10:
        raise InsufficientOverlapError('reference frame has fewer than 10 contact pixels')
    if n_samples is not None and n_samples < len(points):
        pick = (rng if rng is not None else np.random.default_rng()).choice(len(points), n_samples,
                                                                            replace=False)
        points, rows, cols = points[pick], rows[pick], cols[pick]
    ref_normals = ref.normal[rows, cols]
    jacobian = _jacobian(ref.normal, points, rows, cols, mm)
    cur_maps = _padded(np.dstack([cur.contact, cur.normal]))
    for i in range(max_iters):
        _, x, y, inside = _project(T, points, ref.height.shape, mm)
        sampled = _sample(cur_maps, x, y)
        shared = (sampled[:, 0] > .5) & inside
        if shared.sum() < 10:
            raise InsufficientOverlapError()
        residual = (sampled[:, 1:] @ np.linalg.inv(T[:3, :3]).T - ref_normals)[shared].reshape(-1)
        step = lstsq(jacobian[shared].reshape(-1, 5), residual, lapack_driver='gelsy')[0]
        update = np.eye(4)
        update[:3, :3] = Rotation.from_euler('xyz', step[:3]).as_matrix()
        update[:2, 3] = step[3:]
        T = T @ np.linalg.inv(update)
        T[2, 3] = 0.
        if np.linalg.norm(step[:3]) < 1e-4 and np.linalg.norm(step[3:]) < 1e-5 and i > 5:
            break
    moved, x, y, inside = _project(T, points, ref.height.shape, mm)
    contact = (_sample(cur_maps, x, y)[:, 0] > .5) & inside
    if not contact.any():
        raise InsufficientOverlapError()
    T[2, 3] = np.mean(_sample(_padded(cur.height), x, y)[contact] * mm / 1000. - moved[contact, 2])
    return T


def _sobel5(image, axis):
    # cv2.Sobel(ksize=5, scale=2**-7) is a separable correlation with a reflect-101 (scipy 'mirror') border.
    out = correlate1d(image, _DERIVATIVE, axis=axis, mode='mirror')
    return correlate1d(out, _SMOOTH, axis=1 - axis, mode='mirror') * 2. ** -7


def _padded(image):
    # One pixel of zeros, so samples beyond the edge blend toward zero like cv2.remap's constant border.
    image = np.asarray(image, dtype=float)
    return np.pad(image, ((1, 1), (1, 1)) + ((0, 0),) * (image.ndim - 2))


def _sample(padded, x, y):
    # Bilinear samples of a _padded image at original pixel coordinates (x = column, y = row);
    # all channels at once, which is what makes this faster than per-channel map_coordinates.
    x = np.clip(np.asarray(x, dtype=float) + 1, 0, padded.shape[1] - 1)
    y = np.clip(np.asarray(y, dtype=float) + 1, 0, padded.shape[0] - 1)
    x0 = np.minimum(x.astype(int), padded.shape[1] - 2)
    y0 = np.minimum(y.astype(int), padded.shape[0] - 2)
    fx, fy = x - x0, y - y0
    if padded.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((1 - fy) * ((1 - fx) * padded[y0, x0] + fx * padded[y0, x0 + 1])
            + fy * ((1 - fx) * padded[y0 + 1, x0] + fx * padded[y0 + 1, x0 + 1]))


def _pointcloud(height, mm):
    rows, cols = np.mgrid[:height.shape[0], :height.shape[1]].astype(float)
    cols -= height.shape[1] / 2 - .5
    rows -= height.shape[0] / 2 - .5
    return (np.stack((cols, rows, height), axis=-1) * mm / 1000.).reshape(-1, 3)


def _project(T, points, shape, mm):
    moved = points @ T[:3, :3].T + T[:3, 3]
    x = moved[:, 0] * 1000. / mm + shape[1] / 2 - .5
    y = moved[:, 1] * 1000. / mm + shape[0] / 2 - .5
    inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    return moved, x, y, inside


def _jacobian(normal, points, rows, cols, mm):
    """d(residual)/d(rx, ry, rz, tx, ty) per sampled point: [n,3,5] (upstream get_J)."""
    dn_dx = np.stack([_sobel5(normal[..., c], 1) for c in range(3)], -1)[rows, cols] * 1000. / mm
    dn_dy = np.stack([_sobel5(normal[..., c], 0) for c in range(3)], -1)[rows, cols] * 1000. / mm
    normal_gradient = np.stack([dn_dx, dn_dy], axis=-1)
    x, y, z = points.T
    zero, one = np.zeros_like(x), np.ones_like(x)
    pixel_motion = np.stack([np.stack([zero, z, -y, one, zero], -1),
                             np.stack([-z, zero, x, zero, one], -1)], axis=1)
    n = normal[rows, cols]
    rotation = np.zeros((len(x), 3, 5))
    rotation[:, 0, 1], rotation[:, 0, 2] = n[:, 2], -n[:, 1]
    rotation[:, 1, 0], rotation[:, 1, 2] = -n[:, 2], n[:, 0]
    rotation[:, 2, 0], rotation[:, 2, 1] = n[:, 1], -n[:, 0]
    return normal_gradient @ pixel_motion - rotation
