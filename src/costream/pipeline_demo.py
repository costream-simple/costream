"""Offline CoStream example: synthetic rollout tracks and gel frames through StageRunner."""

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from . import synthetic
from .predictive import lift_to_task_frame, resolve_metric_scale, select_tracks
from .reactive import KeyframeTracker, SlipCompensator, TactilePipeline
from .runtime import StageRunner
from .specs import CompositionSpec, StageSpec

FRAMES = 17
DT = .125
SLIP_MM = .4
SLIP_DEG = 3.


def _angle_deg(rotation):
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


def run(seed=0):
    # Predictive: a box carried 6 cm along task x while yawing 0.35 rad, reconstructed at 1/4 scale.
    W_T_C0 = synthetic.camera_pose()
    W_T_I = synthetic.pose((.5, 0., .02), (0., 0., .3))
    W_T_O = [W_T_I @ synthetic.pose((.06 * s, 0., .03), (0., 0., .35 * s)) for s in np.linspace(0., 1., FRAMES)]
    tracks, measured = synthetic.rigid_tracks(synthetic.box_points(), W_T_O, W_T_C0=W_T_C0, scale=4.)
    scale, _ = resolve_metric_scale(tracks.depth0, measured)
    idx, _ = select_tracks(tracks)
    truth = [np.linalg.inv(W_T_I) @ T for T in W_T_O]
    trajectory, _ = lift_to_task_frame(tracks, idx, scale=scale, W_T_C0=W_T_C0, W_T_I=W_T_I, times=DT,
                                       rotation='kabsch', R_task=truth[0][:3, :3])

    # Reactive: the held part slips SLIP_MM along sensor x and SLIP_DEG about sensor z over the stage.
    E_T_S = synthetic.pose((0., .04, .1), (np.pi / 2, 0., 0.))
    tracker = KeyframeTracker(synthetic.PassthroughReconstructor(), mm_per_pixel=synthetic.MM_PER_PIXEL,
                              seed=seed)
    tactile = TactilePipeline(tracker, SlipCompensator(E_T_S))
    runner = StageRunner(StageSpec('transfer', 'carry the part to the fixture', 'compliant_insertion',
                                   'rigid_insertion'),
                         CompositionSpec(owned_axes=(1, 1, 1, 1, 1, 1), max_translation=.01),
                         W_T_I, trajectory)

    records, resets, sample = [], 0, None
    for k, time in enumerate(trajectory.times):
        fraction = k / (FRAMES - 1)
        image = synthetic.tactile_maps(dx_mm=SLIP_MM * fraction, yaw=np.radians(SLIP_DEG) * fraction)
        sample = tactile.start(image, time) if k == 0 else tactile.step(image, time)
        resets += int(tactile.last_result.keyframe_reset)
        packet = runner.tick(time, sample, (0., 0., 5.))
        records.append({'time': float(time), 'tracker_status': tactile.last_result.status,
                        'tactile_status': packet.tactile_status, 'reason': packet.reason,
                        'command': None if packet.command is None else packet.command.tolist()})
    if sample is None:
        raise RuntimeError('tactile tracking did not produce a final sample')

    slip = synthetic.pose((SLIP_MM / 1000, 0., 0.), (0., 0., np.radians(SLIP_DEG)))
    expected = E_T_S @ np.linalg.inv(slip) @ np.linalg.inv(E_T_S)
    # Correction actually commanded after StageRunner's masking and bounds.
    commanded = (np.linalg.inv(W_T_I @ trajectory.sample(trajectory.times[-1]))
                 @ np.asarray(records[-1]['command']))
    summary = {
        'scale': scale,
        'trajectory_max_error_mm': float(max(np.linalg.norm(p[:3, 3] - t[:3, 3])
                                             for p, t in zip(trajectory.poses, truth)) * 1000),
        'trajectory_max_error_deg': max(_angle_deg(p[:3, :3].T @ t[:3, :3]) for p, t in zip(trajectory.poses, truth)),
        'tactile_correction_error_mm': float(np.linalg.norm(sample.correction[:3, 3] - expected[:3, 3]) * 1000),
        'tactile_correction_error_deg': _angle_deg(sample.correction[:3, :3].T @ expected[:3, :3]),
        'commanded_correction_error_mm': float(np.linalg.norm(commanded[:3, 3] - expected[:3, 3]) * 1000),
        'commanded_correction_error_deg': _angle_deg(commanded[:3, :3].T @ expected[:3, :3]),
        'keyframe_resets': resets,
    }
    return {'provenance': 'synthetic rigid-box rollout and Gaussian-bump gel frames (costream.synthetic)',
            'summary': summary, 'records': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0, help='seed for NormalFlow pixel sampling')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.seed), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
