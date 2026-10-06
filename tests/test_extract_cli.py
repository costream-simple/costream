import json

import numpy as np
import pytest

from costream import synthetic
from costream.predictive.cli import main
from costream.trajectory import Trajectory

W_T_C0 = synthetic.camera_pose()
W_T_I = synthetic.pose((.5, 0., .02), (0., 0., .3))


@pytest.fixture
def rollout(tmp_path):
    W_T_O = [W_T_I @ synthetic.pose((.06 * s, 0., .03), (0., 0., .35 * s)) for s in np.linspace(0., 1., 5)]
    tracks, measured = synthetic.rigid_tracks(synthetic.box_points(), W_T_O, W_T_C0=W_T_C0, scale=4.)
    c2w0 = synthetic.pose((.1, .2, -.3), (.2, .1, 0.))  # SpaTrackerV2's own reconstruction world
    T = len(tracks.points)
    np.savez(tmp_path / 'result.npz',
             coords=tracks.points @ c2w0[:3, :3].T + c2w0[:3, 3],
             visibs=tracks.visible.astype(float),
             depths=np.repeat(tracks.depth0[None], T, axis=0),
             intrinsics=np.repeat(synthetic.camera_intrinsics()[None], T, axis=0),
             extrinsics=np.repeat(np.linalg.inv(c2w0)[None], T, axis=0))
    np.save(tmp_path / 'depth.npy', np.rint(measured * 1000).astype(np.uint16))
    truth = [np.linalg.inv(W_T_I) @ W_T_Ot for W_T_Ot in W_T_O]
    for name, value in (('cam.json', W_T_C0), ('anchor.json', W_T_I), ('rot.json', truth[0][:3, :3])):
        (tmp_path / name).write_text(json.dumps(value.tolist()))
    return tmp_path, truth


def _args(directory, *extra):
    return ['--tracks', str(directory / 'result.npz'), '--measured-depth', str(directory / 'depth.npy'),
            '--depth-scale', '0.001', '--dt', '0.25', '--W-T-C0', str(directory / 'cam.json'),
            '--W-T-I', str(directory / 'anchor.json'), '--output', str(directory / 'traj.json'), *extra]


def test_extract_round_trips_through_trajectory_loader(rollout, capsys):
    directory, truth = rollout
    main(_args(directory, '--rotation', 'kabsch', '--R-task', str(directory / 'rot.json')))
    trajectory = Trajectory.from_json(directory / 'traj.json')
    np.testing.assert_allclose(trajectory.poses, truth, atol=1e-9)
    np.testing.assert_allclose(trajectory.times, np.arange(5) * .25)
    data = json.loads((directory / 'traj.json').read_text())
    assert data['scale']['scale'] == pytest.approx(4.) and data['rotation_mode'] == 'kabsch'
    assert 'wrote' in capsys.readouterr().out


def test_extract_with_mask_and_waypoints(rollout):
    directory, truth = rollout
    np.save(directory / 'mask.npy', np.ones((240, 320), np.uint8))
    (directory / 'wp.json').write_text(json.dumps([truth[-1][:3, 3].tolist()]))
    main(_args(directory, '--mask', str(directory / 'mask.npy'), '--waypoints', str(directory / 'wp.json')))
    data = json.loads((directory / 'traj.json').read_text())
    assert data['track_selection']['n_tracks_in_mask'] == 8
    assert data['deviation_from_waypoints']['max_closest_approach_mm'] < 1e-6


def test_extract_refuses_to_overwrite(rollout):
    directory, _ = rollout
    (directory / 'traj.json').write_text('{}')
    with pytest.raises(SystemExit) as raised:
        main(_args(directory))
    assert raised.value.code == 2 and (directory / 'traj.json').read_text() == '{}'


def test_extract_requires_the_anchor(rollout):
    directory, _ = rollout
    args = _args(directory)
    position = args.index('--W-T-I')
    del args[position:position + 2]
    with pytest.raises(SystemExit):
        main(args)


@pytest.mark.parametrize('dt', ['0', '-0.1'])
def test_extract_rejects_nonpositive_dt(rollout, dt):
    directory, _ = rollout
    args = _args(directory)
    args[args.index('--dt') + 1] = dt
    with pytest.raises(SystemExit) as raised:
        main(args)
    assert raised.value.code == 2 and not (directory / 'traj.json').exists()


def test_extract_requires_an_explicit_depth_scale(rollout):
    directory, _ = rollout
    args = _args(directory)
    position = args.index('--depth-scale')
    del args[position:position + 2]
    with pytest.raises(SystemExit) as raised:
        main(args)
    assert raised.value.code == 2 and not (directory / 'traj.json').exists()
