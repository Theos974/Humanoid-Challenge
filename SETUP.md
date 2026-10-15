# Setup

Tested on Windows 11 + WSL2 Ubuntu 24.04, RTX 4060 laptop GPU (8 GB), Python 3.12.

## Simulator + policy (LeRobot with LIBERO and SmolVLA)

```bash
git clone https://github.com/huggingface/lerobot && cd lerobot
git checkout 8c920c4270460851cedd2737657584586d3dc66f
uv venv ~/humanoid/.venv --python 3.12 && source ~/humanoid/.venv/bin/activate
uv pip install "cmake<4"
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install -e ".[smolvla,libero]"
echo N | python -c "import libero.libero"      # answers LIBERO's first-run question once
export MUJOCO_GL=egl
```

## Hand tracking (separate environment: MediaPipe pins its own numpy/protobuf)

```bash
uv venv ~/humanoid/.venv-vision --python 3.12 && source ~/humanoid/.venv-vision/bin/activate
uv pip install mediapipe==1.0.1 opencv-python numpy
mkdir -p ~/humanoid/models && wget -O ~/humanoid/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

On a machine without libGLESv2, MediaPipe fails to import; `apt-get download libgles2 libglvnd0` and pointing
`LD_LIBRARY_PATH` at the extracted `usr/lib/x86_64-linux-gnu` works without root.

## Paths

| variable | default | contents |
|---|---|---|
| `P2R_DATA` | `~/humanoid/data` | phone clips, one folder per session |
| `P2R_WORK` | `~/humanoid/work` | tracks, demos, datasets, checkpoints, eval results |

`data/` in this repo holds the processed clip records (hand paths, grasp/release frames, bowl start), so everything
from `make_demos.py` onwards runs without the raw videos. `splits/` holds the exact held-out layouts and clip
selections behind the reported numbers; `results/` holds every evaluation file the README quotes.
