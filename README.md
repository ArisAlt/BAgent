# BAgent
# version: 0.7.1
# path: README.md



A toolkit for automating EVE Online interactions. The project includes a Gym environment, UI automation modules, and utilities for OCR and computer vision.

## Installation

```bash
pip install -r requirements.txt
```

The repository ships with `sitecustomize.py`, which automatically adds
`src/` to `PYTHONPATH`. Convenience wrappers (`env.py`, `bot_core.py`,
`roi_capture.py`) mirror their counterparts under `src/` for easy imports.
These top-level modules simply re-export everything from the `src` package so
imports work whether you run scripts from the repository root or from the
`tests/` directory. They also provide lightweight fallbacks for optional
dependencies like OpenCV when running the unit tests.

Modules such as `roi_capture` now insert their folder into `sys.path` before
importing sibling modules. Similarly, `src/agent.py` adds the project root to
`sys.path` so it can import `pre_train_data` even when scripts are launched from
within the `src` directory. These tweaks prevent `ModuleNotFoundError` when the
wrappers are used without modifying `PYTHONPATH`. The `AIPilot` class now
imports `OcrEngine` directly from `src/ocr.py`, ensuring the default OCR
provider is available without requiring callers to pass an instance explicitly.
Recent updates switched intra-package imports to explicit relative form
(`from .ocr import OcrEngine`, etc.). If you encounter `ModuleNotFoundError`
for modules like `ocr`, ensure you're running commands from the repository root
or that `src/` is on `PYTHONPATH`. The directory now ships with a minimal
`src/__init__.py` so the package is recognized even on Python versions that do
not support implicit namespace packages.

`Ui.capture` now always captures the full window and crops to the active
region if one is loaded. Screen capture first attempts an `mss` grab of the
target window bounds before falling back to the `PrintWindow` GDI path and,
if required, `pyautogui` or `ImageGrab`. Logs continue to note which
strategy succeeded. Installing requirements now pulls in the additional `mss`
dependency (and `pywin32` for Windows fallbacks). The `requests` library is
also bundled for the LM Studio planning client.

Run tests with:

```bash
pytest -q
```

## Object Detection Pipeline

Real-time UI recognition now uses a YOLOv8 ONNX model executed through
`onnxruntime`. The helper in `src/detector.py` loads the network once and
serves detections to `CvEngine`. To enable the detector:

1. Download a YOLOv8 ONNX checkpoint (e.g., `yolov8n.onnx`) from the
   [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases)
   and place it at `models/yolov8n.onnx` (or update the path in
   `src/config/agent_config.yaml` under the `detector.model_path` setting).
2. Install the updated requirements (which now include `onnxruntime`).
3. Annotate `detect`-type ROIs with detector class labels in
   `src/regions.yaml` or override them via the `detector.roi_map`
   configuration. Each class ID must be mapped to a semantic label in the
   `detector.class_names` section.

`EveEnv` queries the detector for each ROI and consumes bounding boxes and
confidence scores instead of template scales. Reward shaping now leverages
the configured detector labels (`detector.reward_labels`) so you can tune
which detections correspond to mining activity, hostiles, and other events.

## Quick Start

Clone the repository and install the required packages:

```bash
git clone <repo-url>
cd BAgent
pip install -r requirements.txt
```

Record demonstrations with `data_recorder.py`. Pass `--manual` to capture your
own actions; omit the flag for automatic playback. Use `--log` to specify the
JSONL output (defaults to `logs/demonstrations/log_<timestamp>.jsonl`). The
window title is built from `src/config/pilot_name.txt`; override it with
`--window-title`.

```bash
python data_recorder.py --manual --log logs/demonstrations/my_log.jsonl  --window-title "EVE - MyCharacter"
```

Train a behavior cloning model from the recorded file:

```bash
python pre_train_data.py --demos logs/demonstrations/demo.jsonl --out bc_model.pt
```

Launch reinforcement learning or inference using `run_start.py`:

```bash
python run_start.py --train --bc_model bc_model.pt --timesteps 50000
```

## LLM Planning Mode

EveBot can delegate planning to a local LM Studio instance. When LLM planning
is enabled the bot gathers the current observation vector, OCR text snippet,
and the most recent YOLO detections and posts the structured payload to
`src/llm_client.py`. The perception blob now includes a `status` section that
summarises live mining telemetry—cargo hold percentage, which module slots are
cycling, whether a target is locked, whether hostiles were detected on the
current frame, and the last environment reward (all values are clipped for
JSON safety). A new `capabilities` section mirrors the UI affordances so the
planner knows which commands and ROI identifiers are valid. The LM Studio
endpoint (default `http://localhost:1234/v1/chat/completions`) is configurable
in `src/config/agent_config.yaml` under the `llm` section.

`DEFAULT_SYSTEM_PROMPT` now inlines a concise schema harvested directly from
`Ui.execute`, describing every supported command (click, move, drag, hotkey,
type, scroll, sleep, switch_region, sequence, noop, etc.) and their parameters.
The perception payload's `capabilities.commands` mirrors this schema in JSON so
the LLM can validate its own output. Custom planners can override the prompt by
setting `llm.system_prompt` in the config file; any override will replace the
default while still receiving the same capability metadata. Responses must be a
JSON object with an `actions` array—each entry is forwarded verbatim to
`Ui.execute`, and helper fields like `sleep_after` remain supported for
post-action pauses.

1. Start LM Studio with a chat-completion model and enable streaming or JSON
   replies.
2. Enable planning via `llm.enabled: true` in the config or pass
   `--llm-planning` when running `python -m src.bot_core`. Use
   `--no-llm-planning` to force the heuristic fallback.
3. Adjust `endpoint`, `plan_path`, `model`, `temperature`, `timeout`, or the
   `system_prompt` in the same config block to match your setup.
4. Launch the GUI. The footer displays whether LLM planning is active and the
   log records the high-level actions (`click`, `hotkey`, etc.) returned by the
   model. If the server cannot be reached, EveBot automatically falls back to
   the existing mining routine.

Example response expected from LM Studio:

```json
{
  "actions": [
    {"type": "click", "x": 1250, "y": 740},
    {"type": "wait", "duration": 0.5},
    {"type": "hotkey", "keys": ["CTRL", "F"], "interval": 0.05},
    {"type": "type", "text": "Veldspar"},
    {"type": "sleep", "duration": 1.0}
  ]
}
```

Commands that the parser cannot interpret are ignored, and the heuristics take
over for that tick.

### Debug Logging

Set the environment variable `LOG_LEVEL` or pass `--log-level` to `run_start.py`
to control verbosity. Logs follow the format `[HH:MM:SS] LEVEL - message`.

All modules now import the logger via `from src.logger import get_logger` or its
relative form within `src/` (e.g. `from .logger import get_logger`). This
ensures the package can be executed both from the repository root and the
`src/` directory without `ModuleNotFoundError`.

Additional integration tests for ROI/UI via the GUI and CLI can be executed with:

```bash
pytest tests/test_gui_cli_integration.py -q
```

### File Versioning

Each Python source file begins with comments specifying its `version` and
`path`. This metadata helps track changes across modules and corresponds to the
entries in `Scaffold.md`.
## Human-in-the-loop Modes

During runtime press **F9** for Auto, **F10** for Manual, and **F11** for Assistive mode. In Assistive mode press **F12** to execute the suggested action. Pass `--llm-planning` or `--no-llm-planning` on the `python -m src.bot_core` command line to override the configuration toggle for LM Studio plans.

## Behavior Cloning Pretraining

1. Record demonstrations (frames + state logs saved under
   `logs/demonstrations/` by default):

```bash
python data_recorder.py --log logs/demonstrations/demo.jsonl
```

2. Train the BC model. `pre_train_data.py` uses scikit-learn's
   `StandardScaler` and `train_test_split` to preprocess state features and
   create a validation split:

```bash
python pre_train_data.py --demos logs/demonstrations/demo.jsonl --out bc_model.pt
```

The script standardizes observations, splits the data into train/validation
sets and trains a PyTorch model. Both the model weights and fitted scaler are
saved for later inference.

3. Fine-tune with PPO (optional `--bc_model`):

```bash
python run_start.py --train --bc_model bc_model.pt --timesteps 50000
```

4. Train a quick MLP-based policy directly from the demonstration log:

```python
from src.agent import AIPilot
pilot = AIPilot()
pilot.train_bc_from_data('logs/demonstrations/demo.jsonl', 'bc_clf.joblib')
action_idx = pilot.load_and_predict({'obs': [0]*pilot.env.observation_space.shape[0]})
```

### Using `data_recorder.py`

The recorder listens for mouse clicks and key presses while the EVE window is
active. Each event is mapped to an action from the environment's action space
and written to `logs/demonstrations/log_<timestamp>.jsonl` by default. Pass
`--log` to override the file path and `--manual` to collect your own actions;
omit `--manual` for automated playback. The default title comes from
`src/config/pilot_name.txt`; pass `--window-title` to override.

As of version 0.4.4 the recorder stores the pre-action observation in the
pickled buffer so training from `demo_buffer.pkl` matches the JSONL log.

Frame captures are now validated. If `env.ui.capture()` returns `None`,
the step is skipped and a warning is logged. After five consecutive
failures recording aborts to prevent empty data.


### Using `run_start.py`

`run_start.py` acts as the main entry point for training or running the agent.
Passing `--train` starts a PPO training loop, optionally initializing the policy
from a behavior cloning checkpoint via `--bc_model`. Without `--train` the
script loads the latest model under `logs/ppo/` and runs an evaluation episode.

### Running BC Inference

Execute a trained behavior cloning model without launching the GUI:

```bash
python -m src.bot_core --mode bc_inference --bc_model bc_clf.joblib
```

## Session Replay

Visualize recorded demonstrations and compare against a trained model using
`replay_session.py`. The script overlays predicted vs. actual actions and writes
validation metrics to a JSON file:

```bash
python replay_session.py --log logs/demonstrations/demo.jsonl --delay 300 \
    --model bc_model.pt --accuracy-out metrics.json
```

During playback the frame, environment state and predicted action confidence are
displayed. Mismatched predictions are highlighted in red. After exiting, a JSON
file is produced containing overall accuracy, a confusion matrix and per-action
statistics. Press **q** to exit the viewer.


## Replay Correction

Use `replay_correction.py` to modify incorrect actions during playback. Corrected samples are saved with a higher training weight.

```bash
python replay_correction.py --log logs/demonstrations/demo.jsonl --out corrected.jsonl --model bc_model.pt
```

## Mining Helpers

The ``MiningActions`` class implements the sequence of recommended mining
steps defined in *Scaffold.md*. It provides helpers for warping to asteroid
belts, approaching targets and performing human-like idle behaviour. See
``src/mining_actions.py`` for the full list of methods.

## Generating Box Files

Use `generate_box_files.py` to create `.box` files for Tesseract training:

```bash
python generate_box_files.py -i training_texts_dir/images -b training_texts_dir/box
```

If Tesseract is not on your `PATH`, provide the path via `--tesseract-cmd` or
set the `TESSERACT_CMD` environment variable. Windows users can run
`add_tesseract_to_path.bat` with administrator rights to add Tesseract to
`PATH` and set the `TESSERACT_CMD` variable automatically.

### OCR Configuration

`OcrEngine` uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for
text recognition. Once the package is installed no additional configuration is
required:

```python
from ocr import OcrEngine
ocr = OcrEngine()
```
