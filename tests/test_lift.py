import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from costream import synthetic
from costream.predictive.lift import (centroid_motion, deviation_from_waypoints, lift_to_task_frame,
                                      rigid_motion, select_tracks)
from costream.predictive.tracks import TrackSet
from costream.trajectory import Trajectory

W_T_C0 = synthetic.camera_pose()
W_T_I = synthetic.pose((.5, 0., .02), (0., 0., .3))


def _box_motion(n=9, yaw=0., shift=(.06, 0., 0.), scale=4.):
    W_T_O = [W_T_I @ synthetic.pose((shift[0] * s, shift[1] * s, .03 + shift[2] * s), (0., 0., yaw * s))
             for s in np.linspace(0., 1., n)]
    tracks, _ = synthetic.rigid_tracks(synthetic.box_points(), W_T_O, W_T_C0=W_T_C0, scale=scale)
    return tracks, W_T_O


def _tiny(uv0):
    points = np.random.default_rng(0).uniform(.1, .2, size=(2, len(uv0), 3))
    return TrackSet(points, np.ones(points.shape[:2], bool), np.ones((100, 100)), np.asarray(uv0, float))


def test_visibility_threshold_drops_intermittent_tracks():
    tracks, _ = _box_motion()
    visible = tracks.visible.copy()
    visible[1:, 0] = False
    idx, diagnostics = select_tracks(TrackSet(tracks.points, visible, tracks.depth0, tracks.uv0))
    assert 0 not in idx and diagnostics['n_tracks_selected'] == 7


def test_mask_selection_uses_frame0_pixels_and_never_guesses():
    tracks = _tiny([[5, 5], [50, 50], [np.nan, np.nan], [500, 5]])
    mask = np.zeros((100, 100), bool)
    mask[:20, :20] = True
    np.testing.assert_array_equal(select_tracks(tracks, mask=mask)[0], [0])
    idx, diagnostics = select_tracks(tracks, mask=np.ones((100, 100), bool))
    np.testing.assert_array_equal(idx, [0, 1])
    assert diagnostics['n_tracks_in_mask'] == 2


def test_mask_requires_uv0_and_a_boolean_mask():
    tracks = TrackSet(np.ones((2, 3, 3)), np.ones((2, 3), bool), np.ones((10, 10)))
    with pytest.raises(ValueError, match='uv0'):
        select_tracks(tracks, mask=np.ones((10, 10), bool))
    with pytest.raises(ValueError, match='boolean'):
        select_tracks(_tiny([[1, 1]]), mask=np.ones((100, 100)))


def test_empty_selection_raises():
    with pytest.raises(ValueError, match='no tracks'):
        select_tracks(_tiny([[5, 5]]), mask=np.zeros((100, 100), bool))


def test_centroid_uses_only_visible_points():
    points = np.zeros((2, 3, 3))
    points[:, 2] = np.nan
    points[1, :2] = [[1., 1., 1.], [3., 3., 3.]]
    visible = np.array([[True, True, False], [True, True, False]])
    tracks = TrackSet(points, visible, np.ones((4, 4)))
    np.testing.assert_allclose(centroid_motion(tracks, np.arange(3)), [[0., 0., 0.], [2., 2., 2.]])


def test_centroid_raises_when_no_selected_track_is_visible():
    visible = np.ones((2, 3), bool)
    visible[1] = False
    tracks = TrackSet(np.ones((2, 3, 3)), visible, np.ones((4, 4)))
    with pytest.raises(ValueError, match='frame 1'):
        centroid_motion(tracks, np.arange(3))


def test_rigid_motion_recovers_camera_frame_motion():
    tracks, W_T_O = _box_motion(yaw=.4, shift=(.05, .02, .01), scale=1.)
    motions, diagnostics = rigid_motion(tracks, np.arange(8))
    C0_T_W = np.linalg.inv(W_T_C0)
    for motion, W_T_Ot in zip(motions, W_T_O):
        np.testing.assert_allclose(motion, C0_T_W @ W_T_Ot @ np.linalg.inv(W_T_O[0]) @ W_T_C0, atol=1e-9)
    assert max(diagnostics['rms_residual']) < 1e-12 and diagnostics['n_points'] == [8] * 9


def test_rigid_motion_rejects_collinear_points():
    points = np.zeros((2, 4, 3))
    points[:, :, 0] = np.arange(4)
    points[:, :, 2] = 1.
    tracks = TrackSet(points, np.ones((2, 4), bool), np.ones((4, 4)))
    with pytest.raises(ValueError, match='collinear'):
        rigid_motion(tracks, np.arange(4))


def test_fixed_rotation_lift_matches_world_translation():
    tracks, W_T_O = _box_motion(shift=(.06, .01, .02))
    idx, _ = select_tracks(tracks)
    R_task = Rotation.from_rotvec([0., 0., .2]).as_matrix()
    trajectory, diagnostics = lift_to_task_frame(tracks, idx, scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I,
                                                 times=.1, R_task=R_task)
    truth = [np.linalg.inv(W_T_I) @ T for T in W_T_O]
    np.testing.assert_allclose(trajectory.poses[:, :3, 3], [T[:3, 3] for T in truth], atol=1e-12)
    np.testing.assert_allclose(trajectory.poses[:, :3, :3], np.broadcast_to(R_task, (9, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(trajectory.times, np.arange(9) * .1)
    assert diagnostics == {'rotation_mode': 'fixed'}


def test_fixed_lift_holds_rotation_even_when_the_object_yaws():
    tracks, _ = _box_motion(yaw=.35)
    trajectory, _ = lift_to_task_frame(tracks, select_tracks(tracks)[0], scale=4., W_T_C0=W_T_C0,
                                       W_T_I=W_T_I, times=.1)
    np.testing.assert_allclose(trajectory.poses[-1][:3, :3], np.eye(3), atol=1e-12)


def test_kabsch_lift_recovers_6dof_object_pose():
    tracks, W_T_O = _box_motion(yaw=.35, shift=(.06, 0., .01))
    idx, _ = select_tracks(tracks)
    truth = [np.linalg.inv(W_T_I) @ T for T in W_T_O]
    trajectory, diagnostics = lift_to_task_frame(tracks, idx, scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I,
                                                 times=.1, rotation='kabsch', R_task=truth[0][:3, :3])
    np.testing.assert_allclose(trajectory.poses, truth, atol=1e-9)
    assert max(diagnostics['rms_residual_m']) < 1e-9 and diagnostics['n_points'] == [8] * 9


def test_grasp_offset_composes_on_the_right():
    tracks, _ = _box_motion()
    idx, _ = select_tracks(tracks)
    O_T_E = synthetic.pose((0., 0., .1), (np.pi, 0., 0.))
    plain, _ = lift_to_task_frame(tracks, idx, scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I, times=.1)
    grasped, _ = lift_to_task_frame(tracks, idx, scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I, times=.1, O_T_E=O_T_E)
    np.testing.assert_allclose(grasped.poses, plain.poses @ O_T_E, atol=1e-12)


@pytest.mark.parametrize('times', [0., -.1, np.nan, [0., .1]])
def test_times_must_be_explicit_and_valid(times):
    tracks, _ = _box_motion()
    with pytest.raises(ValueError):
        lift_to_task_frame(tracks, np.arange(8), scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I, times=times)


@pytest.mark.parametrize('kwargs', [{'scale': 0.}, {'scale': np.inf}, {'rotation': 'euler'},
                                    {'R_task': np.diag([1., 1., -1.])}])
def test_rejects_invalid_scale_mode_and_rotation(kwargs):
    tracks, _ = _box_motion()
    arguments = dict(scale=4., W_T_C0=W_T_C0, W_T_I=W_T_I, times=.1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        lift_to_task_frame(tracks, np.arange(8), **arguments)


def test_deviation_from_waypoints_reports_closest_approach_mm():
    trajectory = Trajectory([0, 1, 2], [synthetic.pose((x, 0., 0.)) for x in (0., .01, .02)])
    out = deviation_from_waypoints(trajectory, [[.01, .002, 0.], [.03, 0., 0.]])
    assert out['per_waypoint'][0]['at_frame'] == 1
    assert out['per_waypoint'][0]['closest_approach_mm'] == pytest.approx(2.)
    assert out['max_closest_approach_mm'] == pytest.approx(10.)


def test_mask_must_match_the_tracker_frame_resolution():
    tracks = _tiny([[5, 5]])
    with pytest.raises(ValueError, match='depth0'):
        select_tracks(tracks, mask=np.ones((200, 200), bool))


def test_mask_selection_ignores_tracks_not_visible_in_frame0():
    # An occluded point's projected uv0 can land inside the occluder's mask.
    tracks = _tiny([[5, 5], [6, 6], [7, 7]])
    visible = np.ones((2, 3), bool)
    visible[0, 1] = False
    tracks = TrackSet(tracks.points, visible, tracks.depth0, tracks.uv0)
    mask = np.zeros((100, 100), bool)
    mask[:20, :20] = True
    np.testing.assert_array_equal(select_tracks(tracks, mask=mask, min_visible_frac=0.)[0], [0, 2])


def test_rigid_motion_rejects_too_few_common_points():
    tracks, _ = _box_motion(scale=1.)
    visible = tracks.visible.copy()
    visible[3, 2:] = False
    tracks = TrackSet(tracks.points, visible, tracks.depth0, tracks.uv0)
    with pytest.raises(ValueError, match='frame 3'):
        rigid_motion(tracks, np.arange(8))


def test_noisy_tracks_lift_to_millimetre_accuracy():
    tracks, W_T_O = _box_motion(yaw=.35, shift=(.06, 0., .01), scale=1.)
    points = tracks.points + np.random.default_rng(0).normal(scale=.0005, size=tracks.points.shape)
    tracks = TrackSet(points, tracks.visible, tracks.depth0, tracks.uv0)
    truth = [np.linalg.inv(W_T_I) @ T for T in W_T_O]
    for rotation in ('fixed', 'kabsch'):
        trajectory, diagnostics = lift_to_task_frame(tracks, np.arange(8), scale=1., W_T_C0=W_T_C0, W_T_I=W_T_I,
                                                     times=.1, rotation=rotation, R_task=truth[0][:3, :3])
        errors = np.linalg.norm(trajectory.poses[:, :3, 3] - [T[:3, 3] for T in truth], axis=1)
        assert errors.max() < .002
    angles = [np.degrees(Rotation.from_matrix(p[:3, :3].T @ t[:3, :3]).magnitude())
              for p, t in zip(trajectory.poses, truth)]
    assert max(angles) < 2. and max(diagnostics['rms_residual_m']) < .002
