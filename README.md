# CoStream

**Composing Simple Behaviors for Generalizable Complex Manipulation**
([arXiv:2606.26423](https://arxiv.org/abs/2606.26423) · [project page](https://costream-simple.github.io))

CoStream composes three simple behaviors into one end-effector pose command:

```text
W_T_cmd = W_T_I        @ I_T_traj        @ ΔT
          semantic       predictive        reactive
          (stage anchor) (nominal motion)  (tactile correction)
```

This package is a CPU reference implementation for rebuilding that composition
on your own robot:

| Layer | Module | Input you provide | Output |
|---|---|---|---|
| Composition and guards | `costream.runtime` | anchor, trajectory, latest tactile sample, force | guarded pose packets |
| Predictive | `costream.predictive` | 3D point tracks of a generated rollout, frame-0 depth, camera calibration | `Trajectory` of `I_T_traj` |
| Reactive | `costream.reactive` | gel normal/height/contact maps, sensor calibration | `TactileSample` corrections |
| World model client | `costream.world_model` | an image-to-video server | MP4 rollout |

Not included: the semantic (VLM) layer that produces `W_T_I`, robot drivers or
compliant controllers, world-model weights or servers, and point-tracker or
segmentation models. Core dependencies are NumPy and SciPy only.

## Install and run

Requires Python 3.9 or newer.

```bash
git clone https://github.com/costream-simple/costream.git
cd costream
python -m venv .venv
source .venv/bin/activate
python -m pip install .
costream-demo            # v0.1 composition replay (six JSON packets)
costream-pipeline-demo   # predictive + reactive layers on synthetic data
```

`costream-pipeline-demo` lifts a synthetic rollout (a box seen at 1/4 scale)
to a task-frame trajectory with the Kabsch option. It then tracks a synthetic
0.4 mm / 3° gel slip, feeds both through `StageRunner`, and prints the packets
together with recovery errors against ground truth.

`costream-demo` replays the bundled scenario at `src/costream/data/insertion.json`:

- At 0.00 s, no tactile sample is available.
- At 0.04 and 0.08 s, a correction is clipped to the configured bound.
- At 0.12 and 0.16 s, the sample is stale and the nominal pose is used.
- At 0.20 s, a 21 N observation exceeds the 20 N limit, so execution stops with
  `command: null` and `reason: "force_limit"`.

Run `costream-demo --input scenario.json` to replay your own scenario.

## Frame conventions

All transforms are 4×4 matrices acting on column vectors, in metres and radians.
`A_T_B` maps coordinates in frame B into frame A.

| Symbol | Meaning | Who provides it |
|---|---|---|
| `W_T_C0` | world ← frame-0 camera | your camera calibration |
| `W_T_I` | world ← task frame (stage anchor) | your semantic layer |
| `I_T_traj(t)` | task ← nominal end-effector | `costream.predictive` |
| `O_T_E` | object ← end-effector (grasp offset) | you (default identity) |
| `E_T_S` | end-effector ← tactile sensor | your hand-eye style calibration (required) |
| `S1_T_S0` | maps contact points of gel frame 0 into frame 1 (sensor coordinates) | `costream.reactive.register` |
| `ΔT` | correction in the nominal end-effector body frame | `costream.reactive.SlipCompensator` |

Axis conventions you must match when calibrating:

- **Camera frames** (`C0`) are OpenCV optical frames: x right, y down, z forward
  along the viewing ray. This is what SpaTrackerV2 and `project()` use. A ROS
  `camera_link` frame (x forward) must first be converted to its `*_optical_frame`.
- **The tactile sensor frame** (`S`) has its origin at the gel image centre, with
  x along increasing image columns, y along increasing image rows, and z along
  +height from the reconstruction, all in metres via `mm_per_pixel`. Calibrate
  `E_T_S` to this frame. A mirrored gel image is not a proper rotation of it, so
  un-mirror images before reconstruction.

## Predictive behavior: rollout → `I_T_traj`

1. Generate a rollout from the real seed image (see *World-model interface*).
2. Track 3D points through it, for example with
   [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2). Its code is
   CC BY-NC and is not bundled; only its `result.npz` format is read. Any
   tracker works if you build a `costream.predictive.TrackSet` from its output:
   points `[T,N,3]` in camera-0 coordinates, visibility `[T,N]`, frame-0
   predicted depth, and optional frame-0 pixels.
3. Lift the tracks:

```bash
# 16-bit PNG depth -> .npy (needs Pillow)
python -c "import numpy as np, PIL.Image as I; np.save('depth.npy', np.array(I.open('depth.png')))"
costream-extract --tracks result.npz --measured-depth depth.npy --depth-scale 0.001 \
  --dt 0.0625 --W-T-C0 camera.json --W-T-I anchor.json --mask mask.npy \
  --output trajectory.json
costream-demo --trajectory trajectory.json
```

How the steps work:

- **Scale.** The tracker's reconstruction is up to scale. Frame 0 is the real
  seed image, so the median ratio of measured to predicted depth fixes the
  scale. The ratio's IQR is reported as a confidence signal.
- **Timing.** `--dt` is required. Timing is never inferred from the video's
  frame rate.
- **Rotation.**
  - `--rotation fixed` (default, as in the paper) takes translation from the
    per-frame median of the object tracks and holds orientation at `--R-task`.
  - `--rotation kabsch` additionally fits per-frame rigid motion and reports
    RMS residuals. Use it only when the object tracks are rich and non-collinear.
- **Mask.** `--mask` must be in the tracker's pixel frame (the shape of its
  `depths`; SpaTrackerV2 resizes to a 336 px long side). Resample a
  camera-resolution mask before passing it in. A mismatched shape is rejected.
- **Units.** `--depth-scale` is required: 0.001 for millimetre depth, 1 for
  metres.
- **Validation.** `--waypoints` reports how closely the rollout passes taught
  task-frame waypoints.

The same steps are available in Python: `load_spatracker_npz`,
`resolve_metric_scale`, `select_tracks`, `lift_to_task_frame` and
`deviation_from_waypoints`.

## Reactive behavior: gel frames → `ΔT`

```python
from costream.adapters.gelsight import GsSdkReconstructor
from costream.reactive import KeyframeTracker, SlipCompensator, TactilePipeline

reconstructor = GsSdkReconstructor('gs_model.pth', background_image, mm_per_pixel=0.0634)
tracker = KeyframeTracker(reconstructor, mm_per_pixel=0.0634)
pipeline = TactilePipeline(tracker, SlipCompensator(E_T_S))
sample = pipeline.start(first_image, time=0.0)   # at grasp; None if there is no contact yet
sample = pipeline.step(image, time=t)            # each control tick; None if no contact or lost
packet = runner.tick(t, sample, force)
```

- **Registration.** `register` is a SciPy port of
  [NormalFlow](https://github.com/rpl-cmu/normalflow) (MIT).
  `tests/test_normalflow_parity.py` checks it against upstream whenever OpenCV
  and `normalflow` are installed.
- **Keyframes.** `KeyframeTracker` re-keyframes when the long-horizon and
  frame-to-frame estimates disagree by more than 3° or 1 mm.
- **Correction.** `SlipCompensator` computes
  `ΔT = E_T_S @ inv(S_now_T_S_start) @ inv(E_T_S)`, which returns the slipped
  object to its nominal pose. `CompositionSpec.owned_axes` and the norm bounds
  in `StageRunner` then limit which components act and by how much.
- **Gains and masking.** `SlipCompensator` gains act about the sensor origin:
  `rotation_gain=0` corrects translation only, with no lever-arm push. By
  contrast, `CompositionSpec.owned_axes` masks the final end-effector-frame
  correction about the end-effector origin. Masking rotation there keeps the
  translation that paired with that rotation, which is up to a few millimetres
  for a gel several centimetres from the TCP. To drop rotation, use
  `rotation_gain=0` instead.
- **Start frame.** If `start()` sees no contact, the first `step()` with
  contact becomes the start frame. Tracking that reports `lost` keeps its
  keyframe and resumes when registration succeeds again.
- **Timing.** One registration takes about 20 ms on a desktop CPU
  (240×320 maps, 5000 samples), and the tracker runs up to two per frame.
  Use `n_samples` to trade accuracy for speed.
- **`mm_per_pixel`** is sensor-specific: millimetres per gel pixel, called
  `ppmm` upstream. It is 0.0634 for the GelSight Mini.
- **Other sensors.** Implement `reconstruct(image) -> TactileMaps`.
- **GelSight setup.**
  [gs_sdk](https://github.com/joehjhuang/gs_sdk) is not on PyPI; install it with
  `python -m pip install git+https://github.com/joehjhuang/gs_sdk.git`.
  Calibrate its model for your gel and capture a no-contact background image.

## Files

```text
src/costream/
    geometry.py specs.py trajectory.py runtime.py   Composition core (v0.1)
    world_model.py demo.py data/insertion.json      World-model client and replay demo (v0.1)
    predictive/  tracks.py scale.py lift.py cli.py  Rollout tracks -> I_T_traj
    reactive/    maps.py registration.py tracker.py compensation.py
                                                    Gel maps -> TactileSample
    adapters/gelsight.py                            Optional gs_sdk reconstructor
    synthetic.py pipeline_demo.py                   Ground-truth inputs and end-to-end demo
tests/                                              CPU tests (synthetic and seeded)
```

## Composition interface

```python
import numpy as np
from costream.geometry import compose

T_anchor = np.eye(4)       # world <- task frame
T_traj = np.eye(4)         # task frame <- nominal end-effector
DeltaT_tactile = np.eye(4) # correction in nominal body coordinates
T_command = compose(T_anchor, T_traj, DeltaT_tactile)
```

The composition equations are:

```text
T_nom     = T_anchor @ T_traj
T_command = T_nom @ DeltaT_tactile
```

Matrices act on column vectors. Positions are metres, rotations are radians,
and time is elapsed stage seconds on one shared clock. The anchor maps task
coordinates into world coordinates. The rightmost correction is expressed in
the nominal end-effector body frame, not directly in world or task coordinates.

`StageSpec` selects the objective, controller profile, guard profile, and
recovery. `CompositionSpec` specifies six binary owned axes
`[tx, ty, tz, rx, ry, rz]`, translation/rotation norm bounds, the maximum
tactile age, and fallback. Pass an anchor and `Trajectory` to `StageRunner`.

At each `tick(time, tactile, force)`, the runner:

1. Uses a copied, latched anchor and samples the nominal trajectory.
2. Projects tactile displacement and rotation-vector components onto the
   owned task axes and clips their Euclidean norms.
3. Converts the correction back into nominal body coordinates and composes it
   by right multiplication. Corrections are not cumulatively integrated.
4. Returns a controller packet, or a terminal stop with no pose command.

`tactile` is the latest `TactileSample` or `None`. Its transform must already
compensate slip; raw slip requires sensor/tool conversion and inversion or
other compensation upstream. Missing/stale tactile input becomes identity.
Invalid transforms, future samples, or backwards timestamps raise errors.

`force` is three xyz values in newtons, or six values containing xyz force
plus torque. Missing/nonfinite force, excess force, or elapsed-stage timeout
stops the runner permanently. Force freshness must be checked upstream. Torque
is checked for finiteness but is not thresholded. Recovery is stop-only.

### Existing world-frame residuals

For a controller that adds world displacement `dp` and left-multiplies
orientation by `Exp(dr)`, use `world_delta_to_body(T_nom, delta6)`.
With nominal orientation `R`, it produces
`DeltaP = R.T @ dp` and `DeltaR = R.T @ Exp(dr) @ R`.
Right multiplication then preserves `p + dp` and `Exp(dr) @ R`.
Simply changing multiplication order without this conversion changes behavior.

### Saved trajectories

`Trajectory` interpolates position linearly and rotation using SLERP, holding
the endpoint poses outside the recorded interval. Its JSON format is:

```json
{
  "times": [0.0, 1.0],
  "I_T_traj": [
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[1,0,0,0], [0,1,0,0], [0,0,1,0.02], [0,0,0,1]]
  ]
}
```

Load it with `Trajectory.from_json("trajectory.json")` or replay it:

```bash
costream-demo --trajectory trajectory.json
```

`costream-extract` writes explicit `times`. For older extraction output
containing `I_T_traj` without `times`, supply the frame interval with `--dt`.
Do not supply both explicit times and `--dt`. Replacing the trajectory keeps
the bundled scenario's short synthetic event stream; use `--input` with your
own scenario for a longer replay.

## World-model interface

Install the optional HTTP client:

```bash
python -m pip install '.[world-model]'
```

The protocol is
`WorldModel.generate(image_path, prompt, *, seed=None) -> VideoPrediction`.
The result contains MP4 bytes in `video` and a server `task_id`.
Implement this method to attach another service or local model.

```python
from pathlib import Path
from costream.world_model import HTTPWorldModel

model = HTTPWorldModel("http://localhost:8000", max_wait=600)
prediction = model.generate(Path("seed.png"), "Insert the part", seed=7)
# prediction.video: MP4 bytes; prediction.task_id: generation job identifier
```

The CLI saves the video and refuses to overwrite an existing file. The output
directory must already exist.

```bash
costream-world-model --server-url http://localhost:8000 \
  --image seed.png --prompt "Insert the part into the fixture" \
  --output prediction.mp4 --seed 7
```

### External server contract

The server must already be running with a loaded/configured model. This package
provides the client interface, not inference weights or a GPU server. Any server
implementing this contract can be used:

| Request | Input | Required response |
|---|---|---|
| `POST /generate` | Multipart file `image`, form `prompt_text`, optional integer `seed` | JSON object with a nonempty string `task_id` |
| `GET /tasks/{task_id}` | Task ID returned by submission | JSON `status`: `queued`, `pending`, `processing`, `completed`, or `failed`; optional `error` |
| `GET /tasks/{task_id}/video` | Completed task ID | Nonempty bytes with content type `video/mp4` |

Construction makes no network calls. The client submits once and never
automatically retries generation. `max_wait` limits polling, `request_timeout`
bounds each HTTP operation (default 30 seconds), and `poll_interval` controls
polling frequency (default 1 second). Polling timeout does not cancel the
server task; retain its ID. HTTP/server failures raise `WorldModelError`;
polling expiration raises `TimeoutError`. The client holds the download in
memory and checks its content type and nonempty body, without decoding it.

Generated video is not a metric trajectory. A point tracker and calibrated
geometry must produce task-relative `I_T_traj` before it can be loaded into the
composition runner. `costream-extract` performs that lift from tracker output
(see *Predictive behavior*). The tracker, segmentation, and any rollout critic
are not bundled.

## Test and build

```bash
python -m pip install '.[test,world-model]'
python -m pytest tests -q
python -m pip install build
python -m build
```

All tests run on CPU, offline, with synthetic seeded data. Without the
`world-model` extra, the HTTP tests are skipped. The NormalFlow parity test
runs only when OpenCV (`.[parity]`) and upstream `normalflow` are importable.

## Citation

```bibtex
@article{chen2026costream,
  title   = {CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation},
  author  = {Chen, Haonan and Ma, Yuxiang and Tian, Stephen and Han, Xiaoshen and Huang, Wenlong and
             Wu, Feiyang and Li, Yunzhu and Wu, Jiajun and Adelson, Edward H. and Du, Yilun},
  journal = {arXiv preprint arXiv:2606.26423},
  year    = {2026}
}
```

License: Apache-2.0. See `LICENSE` and `NOTICE`.
