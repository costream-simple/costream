import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

pytest.importorskip('requests')

from costream.world_model import HTTPWorldModel, WorldModelError


@pytest.fixture
def server():
    state = {'status': 'completed', 'posts': [], 'gets': [], 'video': b'example-video'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, content_type='application/json'):
            payload = json.dumps(value).encode() if content_type == 'application/json' else value
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            state['posts'].append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
            self.reply({'task_id': 'test-task'})

        def do_GET(self):
            state['gets'].append(self.path)
            if self.path.endswith('/video'):
                self.reply(state['video'], 'video/mp4')
            else:
                self.reply({'status': state['status'], 'error': 'generation failed'})

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = Thread(target=http.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{http.server_port}', state
    http.shutdown()
    http.server_close()
    worker.join()


def test_existing_server_protocol(server, tmp_path):
    url, state = server
    image = tmp_path / 'seed.png'
    image.write_bytes(b'input-image')
    model = HTTPWorldModel(url, poll_interval=.01)
    prediction = model.generate(image, 'insert the part', seed=7)
    assert prediction.video == b'example-video'
    assert prediction.task_id == 'test-task'
    assert state['posts'][0][0] == '/generate'
    assert b'name="image"' in state['posts'][0][1]
    assert b'name="prompt_text"' in state['posts'][0][1]
    assert b'insert the part' in state['posts'][0][1]
    assert b'name="seed"' in state['posts'][0][1]
    assert state['gets'] == ['/tasks/test-task', '/tasks/test-task/video']


@pytest.mark.parametrize('status', ['failed', 'unexpected'])
def test_failure_does_not_download(server, tmp_path, status):
    url, state = server
    state['status'] = status
    image = tmp_path / 'seed.png'
    image.write_bytes(b'image')
    with pytest.raises(WorldModelError):
        HTTPWorldModel(url).generate(image, 'insert')
    assert not any(path.endswith('/video') for path in state['gets'])


def test_wait_is_bounded_and_does_not_resubmit(server, tmp_path):
    url, state = server
    state['status'] = 'processing'
    image = tmp_path / 'seed.png'
    image.write_bytes(b'image')
    with pytest.raises(TimeoutError, match='test-task'):
        HTTPWorldModel(url, max_wait=.03, poll_interval=.01).generate(image, 'insert')
    assert len(state['posts']) == 1


def test_invalid_inputs_do_not_submit(server, tmp_path):
    url, state = server
    with pytest.raises(ValueError):
        HTTPWorldModel(url, max_wait=-1)
    with pytest.raises(ValueError):
        HTTPWorldModel('file:///tmp/server')
    with pytest.raises(ValueError):
        HTTPWorldModel(url).generate(tmp_path / 'missing.png', '')
    assert not state['posts']


def test_empty_video_is_not_success(server, tmp_path):
    url, state = server
    state['video'] = b''
    image = tmp_path / 'seed.png'
    image.write_bytes(b'image')
    with pytest.raises(WorldModelError, match='empty'):
        HTTPWorldModel(url).generate(image, 'insert')


def test_cli_writes_video_and_refuses_overwrite(server, tmp_path):
    url, state = server
    image = tmp_path / 'seed.png'
    image.write_bytes(b'image')
    output = tmp_path / 'result.mp4'
    command = [sys.executable, '-m', 'costream.world_model', '--server-url', url,
               '--image', str(image), '--prompt', 'insert', '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b'example-video'
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode == 2
    assert 'already exists' in again.stderr
    assert len(state['posts']) == 1
