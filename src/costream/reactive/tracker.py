"""Long-horizon tactile tracking with keyframe resets (ported from the research estimator)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .maps import TactileMaps, erode_contact
from .registration import InsufficientOverlapError, register


@dataclass(frozen=True)
class TrackResult:
    """transform is S_now_T_S_start (start contact points -> current sensor frame), or None.

    status: 'ok', 'no_contact' (area below threshold; state kept), 'lost'
    (registration failed; state kept), or 'not_started'.
    """

    transform: Optional[np.ndarray]
    status: str
    contact_area: int = 0
    keyframe_reset: bool = False


class KeyframeTracker:
    def __init__(self, reconstructor, *, mm_per_pixel, contact_threshold=50, reset_rotation_deg=3.,
                 reset_translation_mm=1., n_samples=5000, max_iters=50, seed=None):
        for name, value in (('mm_per_pixel', mm_per_pixel), ('reset_rotation_deg', reset_rotation_deg),
                            ('reset_translation_mm', reset_translation_mm)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if contact_threshold < 0:
            raise ValueError('contact_threshold must be nonnegative')
        self.reconstructor = reconstructor
        self.mm_per_pixel = float(mm_per_pixel)
        self.contact_threshold = int(contact_threshold)
        self.reset_rotation_deg = float(reset_rotation_deg)
        self.reset_translation_mm = float(reset_translation_mm)
        self.n_samples = n_samples
        self.max_iters = int(max_iters)
        self._rng = np.random.default_rng(seed)
        self._ref = self._prev = None
        self._ref_T_start = np.eye(4)
        self._prev_T_ref = np.eye(4)

    @property
    def started(self):
        return self._ref is not None

    def start(self, image):
        """Make this frame the start and keyframe; requires contact."""
        maps, area = self._maps(image)
        if area < self.contact_threshold:
            self._ref = self._prev = None
            return TrackResult(None, 'no_contact', area)
        self._ref = self._prev = maps
        self._ref_T_start = np.eye(4)
        self._prev_T_ref = np.eye(4)
        return TrackResult(np.eye(4), 'ok', area)

    def update(self, image):
        if self._ref is None:
            return TrackResult(None, 'not_started')
        cur, area = self._maps(image)
        if area < self.contact_threshold:
            return TrackResult(None, 'no_contact', area)
        cur_T_ref = self._try_register(self._ref, cur, self._prev_T_ref)
        if self._prev is self._ref:
            cur_T_prev = cur_T_ref
        else:
            cur_T_prev = self._try_register(self._prev, cur, np.eye(4))
        reset = cur_T_ref is None or (cur_T_prev is not None and self._inconsistent(cur_T_ref, cur_T_prev))
        if reset:
            if cur_T_prev is None:
                return TrackResult(None, 'lost', area)
            self._ref_T_start = self._prev_T_ref @ self._ref_T_start
            self._ref = self._prev
            cur_T_ref = cur_T_prev
        self._prev = cur
        self._prev_T_ref = cur_T_ref
        return TrackResult(cur_T_ref @ self._ref_T_start, 'ok', area, reset)

    def _maps(self, image):
        maps = self.reconstructor.reconstruct(image)
        maps = TactileMaps(maps.normal, maps.height, erode_contact(maps.contact))
        return maps, int(maps.contact.sum())

    def _try_register(self, ref, cur, init):
        try:
            return register(ref, cur, mm_per_pixel=self.mm_per_pixel, init=init, n_samples=self.n_samples,
                            max_iters=self.max_iters, rng=self._rng)
        except InsufficientOverlapError:
            return None

    def _inconsistent(self, cur_T_ref, cur_T_prev):
        error = np.linalg.inv(cur_T_ref) @ cur_T_prev @ self._prev_T_ref
        angle = np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude())
        return angle > self.reset_rotation_deg or np.linalg.norm(error[:3, 3]) * 1000 > self.reset_translation_mm
