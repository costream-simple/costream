import numpy as np
import pytest

from costream import synthetic
from costream.predictive.tracks import TrackSet, load_spatracker_npz, project


def _points(T=3, N=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-.1, -.1, .5], [.1, .1, .9], size=(T, N, 3))


def _write_npz(path, camera_points, K, c2w0, **extra):
    # SpaTrackerV2 stores coords in its reconstruction world and extrinsics as world -> camera.
    T = len(camera_points)
    world = camera_points @ c2w0[:3, :3].T + c2w0[:3, 3]
    np.savez(path, coords=world, depths=np.full((T, 240, 320), .7),
             intrinsics=np.repeat(K[None], T, axis=0),
             extrinsics=np.repeat(np.linalg.inv(c2w0)[None], T, axis=0), **extra)


def test_trackset_rejects_mismatched_visibility():
    with pytest.raises(ValueError, match='visible'):
        TrackSet(_points(), np.ones((3, 4), bool), np.ones((4, 4)))


def test_trackset_requires_boolean_visibility():
    with pytest.raises(ValueError, match='boolean'):
        TrackSet(_points(), np.ones((3, 5)), np.ones((4, 4)))


def test_trackset_allows_nan_only_for_invisible_points():
    points = _points()
    points[1, 2] = np.nan
    visible = np.ones((3, 5), bool)
    with pytest.raises(ValueError, match='finite'):
        TrackSet(points, visible, np.ones((4, 4)))
    visible[1, 2] = False
    tracks = TrackSet(points, visible, np.ones((4, 4)))
    assert not tracks.points.flags.writeable


def test_project_marks_points_behind_camera_nan():
    K = synthetic.camera_intrinsics()
    uv = project(np.array([[0., 0., 1.], [.1, 0., -1.]]), K)
    np.testing.assert_allclose(uv[0], K[:2, 2])
    assert np.isnan(uv[1]).all()


def test_loader_maps_reconstruction_world_into_camera0(tmp_path):
    camera_points = _points()
    K = synthetic.camera_intrinsics()
    c2w0 = synthetic.pose((.2, -.1, .3), (.1, -.2, .3))
    _write_npz(tmp_path / 'r.npz', camera_points, K, c2w0, visibs=np.ones(camera_points.shape[:2], bool))
    tracks = load_spatracker_npz(tmp_path / 'r.npz')
    np.testing.assert_allclose(tracks.points, camera_points, atol=1e-12)
    np.testing.assert_allclose(tracks.uv0, project(camera_points[0], K), atol=1e-9)
    assert tracks.depth0.shape == (240, 320)


def test_loader_thresholds_probabilities_and_prefers_tracks2d(tmp_path):
    camera_points = _points()
    probabilities = np.full(camera_points.shape[:2] + (1,), .9)
    probabilities[2, 0, 0] = .2
    tracks2d = np.zeros(camera_points.shape[:2] + (3,))
    tracks2d[0, :, :2] = [10., 20.]
    _write_npz(tmp_path / 'r.npz', camera_points, synthetic.camera_intrinsics(), np.eye(4),
               visibs=probabilities, tracks2d=tracks2d)
    tracks = load_spatracker_npz(tmp_path / 'r.npz')
    assert not tracks.visible[2, 0] and tracks.visible.sum() == 3 * 5 - 1
    np.testing.assert_allclose(tracks.uv0, [[10., 20.]] * 5)


def test_loader_names_missing_keys(tmp_path):
    np.savez(tmp_path / 'r.npz', coords=_points())
    with pytest.raises(KeyError, match='visibs'):
        load_spatracker_npz(tmp_path / 'r.npz')


def test_synthetic_rigid_tracks_are_scaled_and_projected():
    W_T_O = [synthetic.pose((.5, 0., .03)), synthetic.pose((.52, 0., .03))]
    tracks, measured = synthetic.rigid_tracks(synthetic.box_points(), W_T_O,
                                              W_T_C0=synthetic.camera_pose(), scale=4.)
    assert tracks.points.shape == (2, 8, 3) and tracks.visible.all()
    np.testing.assert_allclose(tracks.depth0 * 4., measured)
    assert np.isfinite(tracks.uv0).all()
