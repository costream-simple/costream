"""Typed stage/composition interface and deterministic example profiles.

    Numerical examples are inherited from ReKep/control/control_profiles.py.
    They are NOT a calibration for a new robot. No model-generated gains or
    free-form code are accepted by this interface.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StageSpec:
    name: str
    objective: str
    controller_profile: str
    guard_profile: str
    recovery: str = 'stop'

    def __post_init__(self):
        for field in ('name', 'objective', 'controller_profile', 'guard_profile'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f'{field} must be a nonempty string')
        if self.recovery != 'stop':
            raise ValueError('the CPU release supports only stop recovery')


@dataclass(frozen=True)
class CompositionSpec:
    # Ownership is in the task frame; incoming rigid corrections are body-frame.
    owned_axes: tuple = (1, 1, 0, 1, 1, 0)
    max_translation: float = .005
    max_rotation: float = .1
    max_age: float = .08
    residual_frame: str = 'nominal_body'
    fallback: str = 'zero_residual'

    def __post_init__(self):
        axes = np.asarray(self.owned_axes)
        if axes.shape != (6,) or not np.isin(axes, [0, 1]).all():
            raise ValueError('owned_axes must contain six binary values')
        object.__setattr__(self, 'owned_axes', tuple(int(x) for x in axes))
        for name in ('max_translation', 'max_rotation', 'max_age'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
            object.__setattr__(self, name, value)
        if self.max_rotation > math.pi:
            raise ValueError('max_rotation must not exceed pi radians')
        if self.residual_frame != 'nominal_body' or self.fallback != 'zero_residual':
            raise ValueError('supported residual frame/fallback: nominal_body/zero_residual')


@dataclass(frozen=True)
class ControllerSpec:
    stiffness: tuple
    damping: tuple
    max_force: float
    max_stage_s: float


# Selected existing research profiles. Deliberately omit uncalibrated additions.
_STIFFNESS = {
    'free_space': (600, 600, 600, 45, 45, 45),
    'compliant_insertion': (250, 250, 450, 20, 20, 20),
    'high_force_press': (700, 700, 900, 60, 60, 60),
    'gentle_contact': (180, 180, 300, 15, 15, 15),
}
_GUARDS = {
    'delicate_placement': (5., 8.),
    'rigid_insertion': (20., 12.),
    'high_force_locking': (35., 6.),
    'surface_wiping': (15., math.inf),
}


def compile_controller(stage):
    """Resolve symbolic names; unknown names never degrade to loose defaults."""
    if stage.controller_profile not in _STIFFNESS:
        raise ValueError(f'unknown controller profile: {stage.controller_profile}')
    if stage.guard_profile not in _GUARDS:
        raise ValueError(f'unknown guard profile: {stage.guard_profile}')
    stiffness = _STIFFNESS[stage.controller_profile]
    max_force, timeout = _GUARDS[stage.guard_profile]
    return ControllerSpec(stiffness, tuple(round(2 * math.sqrt(k), 6) for k in stiffness),
                          max_force, timeout)
