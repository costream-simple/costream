import json

import pytest

from costream.pipeline_demo import main, run


def test_pipeline_demo_recovers_synthetic_ground_truth():
    out = run(seed=0)
    summary = out['summary']
    assert summary['scale'] == pytest.approx(4.)
    assert summary['trajectory_max_error_mm'] < 1e-6 and summary['trajectory_max_error_deg'] < 1e-6
    assert summary['tactile_correction_error_mm'] < .02 and summary['tactile_correction_error_deg'] < .1
    assert summary['keyframe_resets'] == 0
    assert summary['commanded_correction_error_mm'] < .02 and summary['commanded_correction_error_deg'] < .1
    assert len(out['records']) == 17
    assert all(r['command'] is not None and r['tactile_status'] == 'fresh' for r in out['records'])


def test_pipeline_demo_cli_prints_json(capsys):
    main([])
    printed = json.loads(capsys.readouterr().out)
    assert 'synthetic' in printed['provenance'] and len(printed['records']) == 17
