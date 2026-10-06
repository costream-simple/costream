"""Task-relative trajectory sampling, including saved tracker extraction output."""

import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import transform


class Trajectory:
    def __init__(self, times, poses):
        self.times = np.array(times, dtype=float, copy=True)
        if (self.times.ndim != 1 or not len(self.times)
                or not np.isfinite(self.times).all() or self.times[0] < 0
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError('trajectory times must be finite, nonnegative, strictly increasing')
        self.poses = np.array([transform(pose) for pose in poses])
        if len(self.poses) != len(self.times):
            raise ValueError('trajectory must have one transform per timestamp')
        self.times.setflags(write=False)
        self.poses.setflags(write=False)
        self._slerp = (Slerp(self.times, Rotation.from_matrix(self.poses[:, :3, :3].copy()))
                       if len(self.times) > 1 else None)

    @classmethod
    def from_json(cls, path, dt=None):
        """Read I_T_traj from ReKep extraction; timing is never inferred.

        The extractor records frame poses but no frame period. Supply dt in
        seconds from the actual selected video, or add explicit `times`.
        Extraction supplies translation only with a fixed reference orientation.
        """
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        poses = data['I_T_traj']
        if 'times' in data:
            if dt is not None:
                raise ValueError('supply times or dt, not both')
            times = data['times']
        else:
            if dt is None or not np.isfinite(dt) or dt <= 0:
                raise ValueError('trajectory JSON needs times or an explicit positive dt')
            times = np.arange(len(poses)) * dt
        return cls(times, poses)

    def sample(self, time):
        if not np.isfinite(time) or time < 0:
            raise ValueError('sample time must be finite and nonnegative')
        time = float(np.clip(time, self.times[0], self.times[-1]))
        if self._slerp is None:
            return self.poses[0].copy()
        out = np.eye(4)
        out[:3, 3] = [np.interp(time, self.times, self.poses[:, axis, 3]) for axis in range(3)]
        out[:3, :3] = self._slerp(time).as_matrix()
        return out
