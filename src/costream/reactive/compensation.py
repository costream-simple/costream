"""Turn tracked contact slip into a body-frame correction for StageRunner."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import transform
from ..runtime import TactileSample


class SlipCompensator:
    """ΔT = E_T_S @ G(inv(T)) @ inv(E_T_S) for contact slip T = S_now_T_S_start.

    E_T_S (end-effector <- tactile sensor) must be calibrated; there is no
    default. G scales the translation and rotation vector of the compensation in
    sensor coordinates, so rotation_gain=0 removes rotation without a lever-arm
    push and translation_gain=0 rotates about the sensor origin. StageRunner's
    owned-axis masking and norm bounds act later, about the end-effector origin.
    """

    def __init__(self, E_T_S, *, translation_gain=1., rotation_gain=1.):
        self.E_T_S = transform(E_T_S)
        self.E_T_S.setflags(write=False)
        self._S_T_E = np.linalg.inv(self.E_T_S)
        for name, gain in (('translation_gain', translation_gain), ('rotation_gain', rotation_gain)):
            if not np.isfinite(gain) or gain < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        self.translation_gain = float(translation_gain)
        self.rotation_gain = float(rotation_gain)

    def correction(self, S_now_T_S_start):
        compensation = np.linalg.inv(transform(S_now_T_S_start))
        scaled = np.eye(4)
        scaled[:3, 3] = compensation[:3, 3] * self.translation_gain
        rotvec = Rotation.from_matrix(compensation[:3, :3]).as_rotvec() * self.rotation_gain
        scaled[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        return self.E_T_S @ scaled @ self._S_T_E


class TactilePipeline:
    """Sensor images -> TactileSample, or None when there is no contact or tracking is lost.

    If start() saw no contact, the first step() with contact becomes the start frame.
    """

    def __init__(self, tracker, compensator):
        self.tracker = tracker
        self.compensator = compensator
        self.last_result = None

    def start(self, image, time):
        return self._sample(self.tracker.start(image), time)

    def step(self, image, time):
        result = self.tracker.update(image) if self.tracker.started else self.tracker.start(image)
        return self._sample(result, time)

    def _sample(self, result, time):
        self.last_result = result
        if result.status != 'ok':
            return None
        return TactileSample(time, self.compensator.correction(result.transform))
