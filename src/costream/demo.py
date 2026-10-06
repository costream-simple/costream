"""Run a saved, explicitly sourced CPU scenario and print JSON controller packets."""

import argparse
from dataclasses import asdict
from importlib.resources import files
import json

from .runtime import StageRunner, TactileSample
from .specs import StageSpec, CompositionSpec
from .trajectory import Trajectory


def run(data, trajectory=None):
    if not isinstance(data.get('provenance'), str) or not data['provenance'].strip():
        raise ValueError('scenario needs a nonempty provenance description')
    if trajectory is None:
        trajectory = Trajectory(data['trajectory']['times'], data['trajectory']['I_T_traj'])
    runner = StageRunner(StageSpec(**data['stage']), CompositionSpec(**data['composition']),
                         data['anchor'], trajectory)
    records = []
    latest = None
    for event in data['events']:
        if 'tactile' in event:
            sample = event['tactile']
            latest = None if sample is None else TactileSample(sample['time'], sample['correction'])
        packet = runner.tick(event['time'], latest, event.get('force'))
        controller = asdict(packet.controller)
        # JSON has no infinity; null means no stage timeout in this profile.
        if controller['max_stage_s'] == float('inf'):
            controller['max_stage_s'] = None
        records.append({
            'time': packet.time,
            'command': None if packet.command is None else packet.command.tolist(),
            'tactile_status': packet.tactile_status,
            'reason': packet.reason,
            'controller': controller,
        })
        if packet.reason:
            break
    if not records:
        raise ValueError('scenario must contain at least one event')
    return {'provenance': data['provenance'], 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='scenario JSON; default: bundled synthetic insertion')
    parser.add_argument('--trajectory', help='optional saved I_T_traj JSON from tracker extraction')
    parser.add_argument('--dt', type=float, help='trajectory frame period in seconds (if JSON lacks times)')
    args = parser.parse_args()
    try:
        if args.dt is not None and args.trajectory is None:
            raise ValueError('--dt requires --trajectory')
        if args.input:
            with open(args.input, encoding='utf-8') as stream:
                data = json.load(stream)
        else:
            data = json.loads(files('costream').joinpath('data/insertion.json').read_text(encoding='utf-8'))
        trajectory = Trajectory.from_json(args.trajectory, args.dt) if args.trajectory else None
        if trajectory is not None:
            data['provenance'] += '; trajectory replaced by user-supplied file (not independently verified)'
        print(json.dumps(run(data, trajectory), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
