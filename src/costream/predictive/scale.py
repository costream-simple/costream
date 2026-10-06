"""Metric scale for an up-to-scale reconstruction, from one measured depth frame."""

import numpy as np


def resolve_metric_scale(pred_depth, measured_depth, *, measured_units_to_m=1., min_valid=500):
    """Return (scale, diagnostics) with ``scale * pred_depth`` ≈ metric depth.

    Frame 0 of a generated rollout is the real seed image, so its measured depth
    fixes the scale. The median ratio ignores regions the depth predictor gets
    badly wrong; the ratio IQR is the confidence signal. The measured map is
    resized to the predicted one by nearest neighbour.
    """
    pred = np.asarray(pred_depth, dtype=float)
    measured = np.asarray(measured_depth, dtype=float)
    if pred.ndim != 2 or measured.ndim != 2:
        raise ValueError('depth maps must be 2D')
    units = float(measured_units_to_m)
    if not np.isfinite(units) or units <= 0:
        raise ValueError('measured_units_to_m must be positive and finite')
    if abs(measured.shape[0] / measured.shape[1] - pred.shape[0] / pred.shape[1]) > .01 * pred.shape[0] / pred.shape[1]:
        raise ValueError(f'depth aspect ratios differ (measured {measured.shape}, predicted {pred.shape}); '
                         'the tracker likely cropped the frame, so crop the measured depth the same way')
    if measured.shape != pred.shape:
        rows = np.minimum((np.arange(pred.shape[0]) * measured.shape[0] / pred.shape[0]).astype(int),
                          measured.shape[0] - 1)
        cols = np.minimum((np.arange(pred.shape[1]) * measured.shape[1] / pred.shape[1]).astype(int),
                          measured.shape[1] - 1)
        measured = measured[np.ix_(rows, cols)]
    measured = measured * units
    valid = np.isfinite(pred) & np.isfinite(measured) & (pred > 1e-6) & (measured > 1e-6)
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        raise ValueError(f'only {n_valid} jointly valid depth pixels (need >= {min_valid}); '
                         'the metric scale is unconstrained')
    ratios = measured[valid] / pred[valid]
    scale = float(np.median(ratios))
    q25, q75 = np.percentile(ratios, [25, 75])
    return scale, {'scale': scale, 'n_valid_px': n_valid, 'ratio_iqr': float(q75 - q25),
                   'ratio_iqr_over_median': float((q75 - q25) / scale)}
