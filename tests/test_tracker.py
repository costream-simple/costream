import numpy as np
import pytest

from costream import synthetic
from costream.reactive import tracker as tracker_module
from costream.reactive.registration import InsufficientOverlapError
from costream.reactive.tracker import KeyframeTracker


def _tracker(**kwargs):
    return KeyframeTracker(synthetic.PassthroughReconstructor(), mm_per_pixel=synthetic.MM_PER_PIXEL,
                           seed=0, **kwargs)


class _Script:
    """Stand-in for register(): returns (or raises) the scripted outputs in order."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def __call__(self, ref, cur, **kwargs):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_sliding_contact_accumulates_from_the_start_frame():
    tracker = _tracker()
    assert tracker.start(synthetic.tactile_maps()).status == 'ok'
    for k in range(1, 6):
        result = tracker.update(synthetic.tactile_maps(dx_mm=.1 * k))
    assert result.status == 'ok' and not result.keyframe_reset and result.contact_area > 50
    np.testing.assert_allclose(result.transform[:2, 3] * 1000, [.5, 0.], atol=.02)


def test_no_contact_at_start_and_update_before_start():
    tracker = _tracker()
    result = tracker.start(synthetic.tactile_maps(pressed=False))
    assert result.status == 'no_contact' and result.transform is None and result.contact_area == 0
    assert tracker.update(synthetic.tactile_maps()).status == 'not_started'


def test_contact_loss_keeps_state_and_resumes():
    tracker = _tracker()
    tracker.start(synthetic.tactile_maps())
    assert tracker.update(synthetic.tactile_maps(pressed=False)).status == 'no_contact'
    result = tracker.update(synthetic.tactile_maps(dx_mm=.2))
    assert result.status == 'ok'
    np.testing.assert_allclose(result.transform[:2, 3] * 1000, [.2, 0.], atol=.02)


def test_inconsistent_long_horizon_estimate_resets_keyframe(monkeypatch):
    A = synthetic.pose((.2e-3, 0., 0.))
    B = synthetic.pose((.1e-3, 0., 0.), (0., 0., .01))
    X = synthetic.pose((5e-3, 0., 0.))
    script = _Script(A, X, B)
    monkeypatch.setattr(tracker_module, 'register', script)
    tracker = _tracker()
    tracker.start(synthetic.tactile_maps())
    first = tracker.update(synthetic.tactile_maps())   # previous frame is the keyframe: one registration
    second = tracker.update(synthetic.tactile_maps())  # keyframe (X) and previous (B) disagree by ~5 mm
    np.testing.assert_allclose(first.transform, A)
    assert second.keyframe_reset and script.calls == 3
    np.testing.assert_allclose(second.transform, B @ A, atol=1e-12)


def test_unrecoverable_overlap_reports_lost_and_keeps_keyframe(monkeypatch):
    A = synthetic.pose((.2e-3, 0., 0.))
    script = _Script(A, InsufficientOverlapError(), InsufficientOverlapError(), A @ A, A)
    monkeypatch.setattr(tracker_module, 'register', script)
    tracker = _tracker()
    tracker.start(synthetic.tactile_maps())
    tracker.update(synthetic.tactile_maps())
    assert tracker.update(synthetic.tactile_maps()).status == 'lost'
    result = tracker.update(synthetic.tactile_maps())
    assert result.status == 'ok' and not result.keyframe_reset
    np.testing.assert_allclose(result.transform, A @ A, atol=1e-12)


@pytest.mark.parametrize('kwargs', [{'mm_per_pixel': 0.}, {'contact_threshold': -1},
                                    {'reset_rotation_deg': 0.}, {'reset_translation_mm': np.nan}])
def test_rejects_invalid_settings(kwargs):
    arguments = dict(mm_per_pixel=synthetic.MM_PER_PIXEL)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        KeyframeTracker(synthetic.PassthroughReconstructor(), **arguments)
