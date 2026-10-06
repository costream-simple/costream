import json
import os
from pathlib import Path
import subprocess
import sys


def test_installed_cli_runs_without_lab_paths_or_credentials(tmp_path):
    env = dict(os.environ)
    # Source tests may use PYTHONPATH; artifact tests exercise the installed wheel.
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in env['PYTHONPATH'].split(os.pathsep))
    for key in ('OPENAI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'ROBOFLOW_API_KEY'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-m', 'costream.demo'], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data['provenance'] == 'synthetic CPU example; not a robot trial'
    assert len(data['records']) == 6
    assert {row['tactile_status'] for row in data['records']} == {'missing', 'fresh', 'stale', 'unused'}
    assert data['records'][-1]['command'] is None
    assert data['records'][-1]['reason'] == 'force_limit'


def test_cli_invalid_file_fails_clearly(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{}')
    result = subprocess.run([sys.executable, '-m', 'costream.demo', '--input', str(bad)],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'error:' in result.stderr
    assert 'Traceback' not in result.stderr
