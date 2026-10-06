"""Lift a SpaTrackerV2 rollout to task-frame I_T_traj JSON for Trajectory.from_json."""

import argparse
import json
import os

import numpy as np

from .. import __version__
from .lift import deviation_from_waypoints, lift_to_task_frame, select_tracks
from .scale import resolve_metric_scale
from .tracks import load_spatracker_npz


def _matrix(path, shape):
    with open(path, encoding='utf-8') as stream:
        value = np.asarray(json.load(stream), dtype=float)
    if value.shape != shape:
        raise ValueError(f'{path}: expected a {shape} matrix, got shape {value.shape}')
    return value


def extract(args):
    tracks = load_spatracker_npz(args.tracks)
    scale, scale_diagnostics = resolve_metric_scale(tracks.depth0, np.load(args.measured_depth),
                                                    measured_units_to_m=args.depth_scale)
    mask = None if args.mask is None else np.load(args.mask) > 0
    idx, selection = select_tracks(tracks, mask=mask, min_visible_frac=args.min_visible_frac)
    trajectory, lift = lift_to_task_frame(
        tracks, idx, scale=scale, W_T_C0=_matrix(args.W_T_C0, (4, 4)), W_T_I=_matrix(args.W_T_I, (4, 4)),
        times=args.dt, rotation=args.rotation,
        R_task=None if args.R_task is None else _matrix(args.R_task, (3, 3)),
        O_T_E=None if args.O_T_E is None else _matrix(args.O_T_E, (4, 4)))
    out = {
        'provenance': f'costream-extract {__version__} from {os.path.basename(args.tracks)}',
        'rotation_mode': args.rotation,
        'times': trajectory.times.tolist(),
        'I_T_traj': trajectory.poses.tolist(),
        'scale': scale_diagnostics,
        'track_selection': selection,
        'lift': lift,
    }
    if args.waypoints:
        with open(args.waypoints, encoding='utf-8') as stream:
            out['deviation_from_waypoints'] = deviation_from_waypoints(trajectory, json.load(stream))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', required=True, help='SpaTrackerV2 result.npz')
    parser.add_argument('--measured-depth', required=True, help='measured frame-0 depth as .npy')
    parser.add_argument('--depth-scale', type=float, required=True,
                        help='metres per measured depth unit (0.001 for millimetre depth)')
    parser.add_argument('--dt', type=float, required=True,
                        help='seconds between trajectory samples (never inferred from the video)')
    parser.add_argument('--W-T-C0', required=True, help='JSON 4x4 world <- frame-0 camera')
    parser.add_argument('--W-T-I', required=True, help='JSON 4x4 world <- task frame (stage anchor)')
    parser.add_argument('--mask', help='frame-0 object mask as .npy (nonzero = object)')
    parser.add_argument('--rotation', choices=('fixed', 'kabsch'), default='fixed')
    parser.add_argument('--R-task', help='JSON 3x3 task-frame reference rotation (default identity)')
    parser.add_argument('--O-T-E', help='JSON 4x4 object <- end-effector grasp offset (default identity)')
    parser.add_argument('--waypoints', help='JSON [K,3] taught task-frame waypoints to compare against')
    parser.add_argument('--min-visible-frac', type=float, default=.5)
    parser.add_argument('--output', required=True, help='output JSON; must not already exist')
    args = parser.parse_args(argv)
    try:
        if os.path.exists(args.output):
            raise ValueError(f'refusing to overwrite {args.output}')
        out = extract(args)
        with open(args.output, 'x', encoding='utf-8') as stream:
            json.dump(out, stream, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output}: {len(out['times'])} poses, scale {out['scale']['scale']:.4g}, "
          f"{out['track_selection']['n_tracks_selected']} tracks, rotation {args.rotation}")


if __name__ == '__main__':
    main()
