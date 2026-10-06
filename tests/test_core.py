import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from costream.geometry import compose, world_delta_to_body, transform
from costream.specs import CompositionSpec, StageSpec, compile_controller
from costream.trajectory import Trajectory
from costream.runtime import StageRunner, TactileSample


def pose(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def runner(spec=None, anchor=None):
    return StageRunner(
        StageSpec('insert', 'seat the part', 'compliant_insertion', 'rigid_insertion'),
        spec or CompositionSpec(),
        np.eye(4) if anchor is None else anchor,
        Trajectory([0, 1], [np.eye(4), pose((0, 0, .02))]),
    )


def test_right_multiplication_with_noncommuting_rotations():
    anchor = pose((.4, .1, .2), (0, 0, np.pi / 2))
    nominal = pose((0, 0, .02), (.4, 0, 0))
    residual = pose((.001, 0, 0), (0, .2, 0))
    actual = compose(anchor, nominal, residual)
    expected = anchor @ nominal @ residual
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not np.allclose(actual, residual @ anchor @ nominal)


def test_world_adapter_preserves_legacy_controller_semantics():
    nominal = pose((.5, -.2, .3), (.3, .4, .8))
    delta = np.array([.003, -.002, .001, .1, -.2, .05])
    actual = nominal @ world_delta_to_body(nominal, delta)
    expected = nominal.copy()
    expected[:3, 3] += delta[:3]
    expected[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ nominal[:3, :3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('bad', [np.zeros((4, 4)), np.full((4, 4), np.nan),
                                    np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])])
def test_invalid_transforms_rejected(bad):
    with pytest.raises(ValueError):
        transform(bad)


def test_interpolates_translation_and_rotation_and_holds_endpoints():
    trajectory = Trajectory([0, 2], [np.eye(4), pose((0, 0, .02), (0, 0, np.pi / 2))])
    np.testing.assert_allclose(trajectory.sample(1), pose((0, 0, .01), (0, 0, np.pi / 4)), atol=1e-12)
    np.testing.assert_allclose(trajectory.sample(3), trajectory.sample(2))
    np.testing.assert_allclose(trajectory.sample(0), np.eye(4))


@pytest.mark.parametrize('times', [[0, 0], [1, 0], [0, np.nan], [-1, 0]])
def test_bad_trajectory_timing_rejected(times):
    with pytest.raises(ValueError):
        Trajectory(times, [np.eye(4), np.eye(4)])


def test_extractor_json_requires_explicit_timing(tmp_path):
    path = tmp_path / 'trajectory.json'
    path.write_text(json.dumps({'I_T_traj': [np.eye(4).tolist(), pose((0, 0, .02)).tolist()]}))
    trajectory = Trajectory.from_json(path, dt=.1)
    np.testing.assert_allclose(trajectory.sample(.05)[:3, 3], [0, 0, .01])
    with pytest.raises(ValueError):
        Trajectory.from_json(path)


def test_owned_axes_follow_task_frame_and_bounds_are_norms():
    anchor = pose(rpy=(0, 0, np.pi / 2))
    spec = CompositionSpec(owned_axes=(1, 0, 0, 1, 0, 0), max_translation=.002, max_rotation=.05)
    run = runner(spec, anchor)
    sample = TactileSample(0, pose((.01, .02, .03), (.2, 0, 0)))
    result = run.tick(0, sample, [0, 0, 0])
    np.testing.assert_allclose(result.command[:3, 3], [0, .002, 0], atol=1e-12)
    relative = np.linalg.inv(anchor) @ result.command
    np.testing.assert_allclose(Rotation.from_matrix(relative[:3, :3]).as_rotvec(), [.05, 0, 0], atol=1e-12)


def test_ownership_with_rotated_nominal_uses_task_axes():
    spec = CompositionSpec(owned_axes=(1, 0, 0, 0, 0, 0), max_translation=.1)
    run = runner(spec)
    run.trajectory = Trajectory([0], [pose(rpy=(0, 0, np.pi / 2))])
    # Body +x is task +y and is rejected; body -y is task +x and is retained.
    result = run.tick(0, TactileSample(0, pose((.01, -.02, 0))), [0, 0, 0])
    np.testing.assert_allclose(result.command[:3, 3], [.02, 0, 0], atol=1e-12)


def test_anchor_latched_residual_not_integrated_and_stale_fallback():
    anchor = pose((.4, 0, 0))
    run = runner(CompositionSpec(max_age=.05), anchor)
    anchor[0, 3] = 9
    sample = TactileSample(0, pose((.001, 0, 0)))
    a = run.tick(0, sample, [0, 0, 0])
    b = run.tick(.04, sample, [0, 0, 0])
    c = run.tick(.08, sample, [0, 0, 0])
    assert a.command[0, 3] == pytest.approx(.401)
    assert b.command[0, 3] == pytest.approx(.401)
    assert c.command[0, 3] == pytest.approx(.4)
    assert c.tactile_status == 'stale'


def test_guard_terminates_and_never_emits_another_command():
    run = runner()
    result = run.tick(0, None, [0, 0, 21])
    assert result.command is None
    assert result.reason == 'force_limit'
    assert run.tick(.04, None, [0, 0, 0]).command is None


@pytest.mark.parametrize('force', [None, [np.nan, 0, 0], [0, 0]])
def test_bad_force_stops_execution(force):
    result = runner().tick(0, None, force)
    assert result.command is None
    assert result.reason == 'invalid_force'


def test_future_sample_and_backwards_clock_rejected():
    run = runner()
    with pytest.raises(ValueError):
        run.tick(0, TactileSample(.1, np.eye(4)), [0, 0, 0])
    run.tick(.1, None, [0, 0, 0])
    with pytest.raises(ValueError):
        run.tick(.09, None, [0, 0, 0])


def test_unknown_profiles_and_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        compile_controller(StageSpec('a', 'b', 'unknown', 'rigid_insertion'))
    with pytest.raises(ValueError):
        compile_controller(StageSpec('a', 'b', 'free_space', 'unknown'))
    for kwargs in ({'max_translation': -1}, {'max_rotation': float('nan')},
                   {'owned_axes': [1, 2]}, {'max_age': 0}):
        with pytest.raises(ValueError):
            CompositionSpec(**kwargs)


def test_timeout_stops_execution():
    assert runner().tick(12.1, None, [0, 0, 0]).reason == 'stage_timeout'
