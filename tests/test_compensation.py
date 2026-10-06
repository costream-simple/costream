import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from costream import synthetic
from costream.reactive.compensation import SlipCompensator, TactilePipeline
from costream.reactive.tracker import KeyframeTracker
from costream.runtime import TactileSample

E_T_S = synthetic.pose((0., .04, .1), (np.pi / 2, 0., 0.))


def _rotvec(T):
    return Rotation.from_matrix(T[:3, :3]).as_rotvec()


def test_full_gain_returns_slipped_object_to_its_nominal_world_pose():
    W_T_E = synthetic.pose((.5, .1, .3), (.1, .2, .3))
    S_T_O = synthetic.pose((.001, -.002, .004), (0., .1, 0.))
    slip = synthetic.pose((.4e-3, -.1e-3, 0.), (0., 0., .05))
    correction = SlipCompensator(E_T_S).correction(slip)
    before = W_T_E @ E_T_S @ S_T_O
    after = W_T_E @ correction @ E_T_S @ (slip @ S_T_O)
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_zero_gain_is_identity_and_half_gain_halves_the_correction():
    slip = synthetic.pose((.4e-3, 0., 0.), (0., 0., .05))
    zero = SlipCompensator(E_T_S, translation_gain=0., rotation_gain=0.).correction(slip)
    np.testing.assert_allclose(zero, np.eye(4), atol=1e-15)
    S_T_E = np.linalg.inv(E_T_S)
    full = S_T_E @ SlipCompensator(E_T_S).correction(slip) @ E_T_S
    half = S_T_E @ SlipCompensator(E_T_S, translation_gain=.5, rotation_gain=.5).correction(slip) @ E_T_S
    np.testing.assert_allclose(_rotvec(half), _rotvec(full) / 2, atol=1e-12)
    np.testing.assert_allclose(half[:3, 3], full[:3, 3] / 2, atol=1e-15)


def test_rotation_only_gain_pivots_about_the_sensor_not_the_end_effector():
    # A pure in-plane slip rotation must not drag the contact through an end-effector lever arm.
    slip = synthetic.pose((0., 0., 0.), (0., 0., np.radians(3.)))
    correction = SlipCompensator(E_T_S, translation_gain=0.).correction(slip)
    sensor_origin = E_T_S[:, 3]
    np.testing.assert_allclose(correction @ sensor_origin, sensor_origin, atol=1e-15)
    np.testing.assert_allclose(_rotvec(correction), -E_T_S[:3, :3] @ [0., 0., np.radians(3.)], atol=1e-12)


def test_translation_only_gain_commands_no_rotation():
    slip = synthetic.pose((.4e-3, 0., 0.), (0., 0., np.radians(3.)))
    correction = SlipCompensator(E_T_S, rotation_gain=0.).correction(slip)
    np.testing.assert_allclose(correction[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(correction[:3, 3], E_T_S[:3, :3] @ np.linalg.inv(slip)[:3, 3], atol=1e-15)


@pytest.mark.parametrize('kwargs', [{'translation_gain': -1.}, {'rotation_gain': np.inf}])
def test_rejects_invalid_gains(kwargs):
    with pytest.raises(ValueError, match='gain'):
        SlipCompensator(E_T_S, **kwargs)


def test_rejects_an_invalid_extrinsic():
    with pytest.raises(ValueError):
        SlipCompensator(np.diag([1., 1., -1., 1.]))


def test_pipeline_emits_samples_only_while_tracking():
    tracker = KeyframeTracker(synthetic.PassthroughReconstructor(), mm_per_pixel=synthetic.MM_PER_PIXEL, seed=0)
    pipeline = TactilePipeline(tracker, SlipCompensator(np.eye(4)))
    first = pipeline.start(synthetic.tactile_maps(), 0.)
    assert isinstance(first, TactileSample) and first.time == 0.
    np.testing.assert_allclose(first.correction, np.eye(4), atol=1e-12)
    sample = pipeline.step(synthetic.tactile_maps(dx_mm=.3), .04)
    np.testing.assert_allclose(sample.correction[:3, 3] * 1000, [-.3, 0., 0.], atol=.02)
    assert pipeline.step(synthetic.tactile_maps(pressed=False), .08) is None
    assert pipeline.last_result.status == 'no_contact'


def test_pipeline_starts_tracking_at_first_contact():
    tracker = KeyframeTracker(synthetic.PassthroughReconstructor(), mm_per_pixel=synthetic.MM_PER_PIXEL, seed=0)
    pipeline = TactilePipeline(tracker, SlipCompensator(np.eye(4)))
    assert pipeline.start(synthetic.tactile_maps(pressed=False), 0.) is None and not tracker.started
    first_contact = pipeline.step(synthetic.tactile_maps(dx_mm=.3), .04)
    np.testing.assert_allclose(first_contact.correction, np.eye(4), atol=1e-12)
    assert tracker.started
    later = pipeline.step(synthetic.tactile_maps(dx_mm=.5), .08)
    np.testing.assert_allclose(later.correction[:3, 3] * 1000, [-.2, 0., 0.], atol=.02)
