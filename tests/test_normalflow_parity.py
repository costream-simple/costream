"""Agreement with upstream NormalFlow + OpenCV; skipped unless both are importable."""

import numpy as np
import pytest

from costream import synthetic
from costream.reactive.registration import register

pytest.importorskip('cv2')
upstream = pytest.importorskip('normalflow.registration')


@pytest.mark.parametrize('dx, dy, yaw_deg', [(.3, -.2, 0.), (0., 0., 4.), (.25, .15, -3.), (1.5, -.8, 8.)])
def test_matches_upstream_normalflow(dx, dy, yaw_deg):
    ref = synthetic.tactile_maps()
    cur = synthetic.tactile_maps(dx_mm=dx, dy_mm=dy, yaw=np.radians(yaw_deg))
    arguments = []
    for maps in (ref, cur):
        arguments += [maps.normal.astype(np.float32), maps.contact, maps.height.astype(np.float32)]
    expected = upstream.normalflow(*arguments, np.eye(4), synthetic.MM_PER_PIXEL, n_samples=None)
    actual = register(ref, cur, mm_per_pixel=synthetic.MM_PER_PIXEL, n_samples=None)
    # Upstream runs in float32 with OpenCV's fixed-point remap; measured agreement is < 0.5 um and < 5e-5.
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=2e-6)
    np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=1e-4)
