"""CPU stage runner: compose tactile poses and emit guarded controller packets.

No robot connection or force-responsive pose update is implemented here. A robot
adapter must execute each valid pose using its own calibrated compliant loop.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import compose, transform
from .specs import ControllerSpec, compile_controller


@dataclass(frozen=True)
class TactileSample:
    time: float
    correction: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError('tactile timestamp must be finite and nonnegative')
        matrix = transform(self.correction)
        matrix.setflags(write=False)
        object.__setattr__(self, 'correction', matrix)


@dataclass(frozen=True)
class ControlPacket:
    time: float
    command: Optional[np.ndarray]
    controller: ControllerSpec
    tactile_status: str
    reason: str = ''


class StageRunner:
    def __init__(self, stage, composition, anchor, trajectory):
        self.stage = stage
        self.composition = composition
        self.anchor = transform(anchor)
        self.anchor.setflags(write=False)
        self.trajectory = trajectory
        self.controller = compile_controller(stage)
        self._last_time = -1.
        self._last_sample_time = -1.
        self._stopped = ''

    def tick(self, time, tactile, force):
        """Use elapsed stage seconds and the latest sample (or None).

        Force is world-frame xyz in newtons, or xyz + torque in a six-vector.
        Missing/invalid force halts the stage; missing/stale tactile uses identity.
        A halted runner cannot resume; construct a new stage runner to recover.
        """
        if not np.isfinite(time) or time < 0 or time < self._last_time:
            raise ValueError('controller time must be finite, nonnegative and monotonic')
        if tactile is not None:
            if tactile.time > time or tactile.time < self._last_sample_time:
                raise ValueError('tactile timestamps must be ordered and not in the future')
        self._last_time = time
        if tactile is not None:
            self._last_sample_time = tactile.time
        if not self._stopped:
            self._stopped = self._guard(time, force)
        if self._stopped:
            return ControlPacket(time, None, self.controller, 'unused', self._stopped)

        nominal = self.trajectory.sample(time)
        correction = np.eye(4)
        status = 'missing'
        if tactile is not None:
            status = 'stale' if time - tactile.time > self.composition.max_age else 'fresh'
            if status == 'fresh':
                correction = self._bounded(tactile.correction, nominal[:3, :3])
        return ControlPacket(time, compose(self.anchor, nominal, correction), self.controller, status)

    def _guard(self, time, force):
        try:
            values = np.asarray(force, dtype=float)
        except (TypeError, ValueError):
            return 'invalid_force'
        if values.shape not in ((3,), (6,)) or not np.isfinite(values).all():
            return 'invalid_force'
        if np.linalg.norm(values[:3]) > self.controller.max_force:
            return 'force_limit'
        if time > self.controller.max_stage_s:
            return 'stage_timeout'
        return ''

    def _bounded(self, correction, task_R_body):
        # Rotate displacement/rotvec into task axes, mask and bound there, then
        # return to body coordinates for the paper's right multiplication.
        mask = np.asarray(self.composition.owned_axes)
        translation = (task_R_body @ correction[:3, 3]) * mask[:3]
        rotation = (task_R_body @ Rotation.from_matrix(correction[:3, :3].copy()).as_rotvec()) * mask[3:]
        for vector, limit in ((translation, self.composition.max_translation),
                              (rotation, self.composition.max_rotation)):
            norm = np.linalg.norm(vector)
            if norm > limit:
                vector *= limit / norm
        out = np.eye(4)
        out[:3, 3] = task_R_body.T @ translation
        out[:3, :3] = Rotation.from_rotvec(task_R_body.T @ rotation).as_matrix()
        return out
