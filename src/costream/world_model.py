"""Optional image-to-video interface compatible with the research video server.

No GPU framework is imported. A generated video is not a metric trajectory;
tracking and calibrated extraction must precede control-side trajectory loading.
"""

import argparse
from dataclasses import dataclass
import math
import mimetypes
from pathlib import Path
import time
from typing import Optional, Protocol
from urllib.parse import quote, urlparse


class WorldModelError(RuntimeError):
    """The server request or generation task failed."""


@dataclass(frozen=True)
class VideoPrediction:
    video: bytes
    task_id: str


class WorldModel(Protocol):
    """Implement this method to plug in a local model or another service."""

    def generate(self, image_path: Path, prompt: str,
                 *, seed: Optional[int] = None) -> VideoPrediction:
        ...


class HTTPWorldModel:
    """Client for POST /generate and GET /tasks/{id}[/video].

    The server must already have its model configured. Construction makes no
    network calls. Upload is submitted once, never retried automatically.
    max_wait limits task polling; request_timeout bounds each HTTP operation.
    A timeout does not cancel server-side work (the server has no cancel API).
    """

    def __init__(self, server_url, *, max_wait=600., request_timeout=30.,
                 poll_interval=1.):
        parsed = urlparse(server_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.netloc
                or parsed.query or parsed.fragment or parsed.username):
            raise ValueError('server_url must be an HTTP(S) base URL without credentials/query/fragment')
        for name, value in (('max_wait', max_wait), ('request_timeout', request_timeout),
                            ('poll_interval', poll_interval)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if poll_interval > 60:
            raise ValueError('poll_interval must not exceed 60 seconds')
        try:
            import requests
        except ImportError as exc:
            raise ImportError('Install costream[world-model] for the HTTP adapter') from exc
        self._requests = requests
        self.server_url = server_url.rstrip('/')
        self.max_wait = max_wait
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval

    def _request(self, method, path, **kwargs):
        try:
            response = self._requests.request(method, self.server_url + path,
                                              timeout=self.request_timeout, **kwargs)
            response.raise_for_status()
            return response
        except self._requests.RequestException as exc:
            raise WorldModelError(f'world-model HTTP request failed: {method} {path}') from exc

    @staticmethod
    def _json(response):
        try:
            data = response.json()
        except ValueError as exc:
            raise WorldModelError('world-model server returned invalid JSON') from exc
        if not isinstance(data, dict):
            raise WorldModelError('world-model server must return a JSON object')
        return data

    def generate(self, image_path, prompt, *, seed=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be nonempty')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError('seed must be an integer')
        image_path = Path(image_path)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError('image_path must be a nonempty file')
        data = {'prompt_text': prompt}
        if seed is not None:
            data['seed'] = seed
        with image_path.open('rb') as image:
            mime = mimetypes.guess_type(image_path.name)[0] or 'application/octet-stream'
            response = self._request('POST', '/generate', data=data,
                                     files={'image': (image_path.name, image, mime)})
        task_id = self._json(response).get('task_id')
        if not isinstance(task_id, str) or not task_id:
            raise WorldModelError('world-model server returned no task_id')
        path = '/tasks/' + quote(task_id, safe='')
        deadline = time.monotonic() + self.max_wait
        while time.monotonic() < deadline:
            result = self._json(self._request('GET', path))
            status = result.get('status')
            if status == 'completed':
                response = self._request('GET', path + '/video')
                if not response.content:
                    raise WorldModelError('world-model server returned an empty video')
                if response.headers.get('Content-Type', '').split(';')[0] != 'video/mp4':
                    raise WorldModelError('world-model server did not return video/mp4')
                return VideoPrediction(response.content, task_id)
            if status == 'failed':
                raise WorldModelError(f'world-model task {task_id} failed: {result.get("error")}')
            if status not in ('pending', 'queued', 'processing'):
                raise WorldModelError(f'world-model task {task_id}: unknown status {status!r}')
            time.sleep(min(self.poll_interval, max(0., deadline - time.monotonic())))
        raise TimeoutError(f'world-model task {task_id} exceeded polling budget; server task may still run')


def main():
    parser = argparse.ArgumentParser(description='Generate video using the configured research model server.')
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-wait', type=float, default=600.)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        if output.exists():
            raise ValueError('output already exists; choose a new path')
        if not output.parent.is_dir():
            raise ValueError('output directory must already exist')
        result = HTTPWorldModel(args.server_url, max_wait=args.max_wait).generate(
            args.image, args.prompt, seed=args.seed)
        with output.open('xb') as stream:
            stream.write(result.video)
        print(f'Saved {output} (task {result.task_id})')
    except (ValueError, OSError, ImportError, WorldModelError, TimeoutError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
