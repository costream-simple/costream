import numpy as np
import pytest

from costream.predictive.scale import resolve_metric_scale


def test_median_scale_ignores_outlier_regions():
    pred = np.full((100, 100), .2)
    measured = pred * 4.
    measured[:10] = 9.
    scale, diagnostics = resolve_metric_scale(pred, measured)
    assert scale == pytest.approx(4.)
    assert diagnostics['n_valid_px'] == 10000
    assert diagnostics['ratio_iqr'] == pytest.approx(0.)


def test_millimetre_depth_and_nearest_resize():
    pred = np.full((50, 60), .25)
    measured_mm = np.full((100, 120), 750, dtype=np.uint16)
    scale, _ = resolve_metric_scale(pred, measured_mm, measured_units_to_m=.001)
    assert scale == pytest.approx(3.)


def test_unconstrained_scale_raises():
    with pytest.raises(ValueError, match='unconstrained'):
        resolve_metric_scale(np.full((20, 20), .3), np.zeros((20, 20)))


@pytest.mark.parametrize('units', [0., -1., np.nan])
def test_rejects_invalid_units(units):
    with pytest.raises(ValueError, match='measured_units_to_m'):
        resolve_metric_scale(np.ones((30, 30)), np.ones((30, 30)), measured_units_to_m=units)


def test_rejects_aspect_ratio_mismatch_from_cropping():
    with pytest.raises(ValueError, match='aspect'):
        resolve_metric_scale(np.ones((336, 252)), np.ones((640, 480)).T.T[:, :360])
