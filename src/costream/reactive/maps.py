"""Tactile surface maps and the interface a sensor reconstruction must provide."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import binary_erosion


@dataclass(frozen=True)
class TactileMaps:
    """One gel frame: unit normals [H,W,3], height [H,W] in pixels, contact [H,W] bool."""

    normal: np.ndarray
    height: np.ndarray
    contact: np.ndarray

    def __post_init__(self):
        normal = np.array(self.normal, dtype=float)
        height = np.array(self.height, dtype=float)
        contact = np.array(self.contact)
        if height.ndim != 2 or normal.shape != height.shape + (3,) or contact.shape != height.shape:
            raise ValueError('need normal [H,W,3], height [H,W] and contact [H,W] of one size')
        if contact.dtype != bool:
            raise ValueError('contact must be a boolean mask')
        if not (np.isfinite(normal).all() and np.isfinite(height).all()):
            raise ValueError('normal and height maps must be finite')
        for name, value in (('normal', normal), ('height', height), ('contact', contact)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class TactileReconstructor(Protocol):
    def reconstruct(self, image) -> TactileMaps:
        """Convert one raw sensor image into TactileMaps."""


def gradient_to_normal(gradient):
    """Unit normals [H,W,3] from surface gradients [H,W,2] (dz/dx, dz/dy)."""
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 3 or gradient.shape[2] != 2:
        raise ValueError('gradient must have shape [H,W,2]')
    normal = np.dstack([-gradient, np.ones(gradient.shape[:2])])
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def erode_contact(contact):
    """Erode a contact mask (square kernel of height // 48 pixels) to drop unreliable rim pixels."""
    contact = np.asarray(contact, dtype=bool)
    size = max(contact.shape[0] // 48, 1)
    return binary_erosion(contact, structure=np.ones((size, size)))
