"""GelSight reconstruction through gs_sdk (torch); install it from GitHub, it is not on PyPI."""

import numpy as np

from ..reactive.maps import TactileMaps, gradient_to_normal

_INSTALL = ('GsSdkReconstructor needs gs_sdk: '
            'python -m pip install git+https://github.com/joehjhuang/gs_sdk.git')


class GsSdkReconstructor:
    """TactileReconstructor backed by a calibrated gs_sdk model and a no-contact background image."""

    def __init__(self, model_path, background, *, mm_per_pixel, device='cpu', contact_mode='standard'):
        try:
            from gs_sdk.gs_reconstruct import Reconstructor
        except ImportError as exc:
            raise ImportError(_INSTALL) from exc
        if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
            raise ValueError('mm_per_pixel must be positive and finite')
        self.mm_per_pixel = float(mm_per_pixel)
        self._recon = Reconstructor(str(model_path), contact_mode=contact_mode, device=device)
        self._recon.load_bg(np.asarray(background))

    def reconstruct(self, image):
        gradient, height, contact = self._recon.get_surface_info(np.asarray(image), self.mm_per_pixel)
        return TactileMaps(gradient_to_normal(gradient), height, np.asarray(contact).astype(bool))
