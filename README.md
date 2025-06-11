# BAgent
<!-- version: 0.4.9 | path: README.md -->

A toolkit for automating EVE Online interactions. The project includes a Gym environment, UI automation modules, and utilities for OCR and computer vision.

## Installation

```bash
pip install -r requirements.txt
```

Run tests with:

```bash
pytest -q
```

### Debug Logging

Set the environment variable `LOG_LEVEL` or pass `--log-level` to `run_start.py`
to control verbosity. Logs follow the format `[HH:MM:SS] LEVEL - message`.

Additional integration tests for ROI/UI via the GUI and CLI can be executed with:

```bash
pytest tests/test_gui_cli_integration.py -q
```

## Behavior Cloning Pretraining

1. Record demonstrations (frames + state logs saved under `logs/demonstrations/`):

```bash
python data_recorder.py --manual False
```

2. Train the BC model. `pre_train_data.py` uses scikit-learn's
   `StandardScaler` and `train_test_split` to preprocess state features and
   create a validation split:

```bash
python pre_train_data.py --demos logs/demonstrations/log.jsonl --out bc_model.pt
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
pilot.train_bc_from_data('logs/demonstrations/log.jsonl', 'bc_clf.joblib')
action_idx = pilot.load_and_predict({'obs': [0]*pilot.env.observation_space.shape[0]})
```

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
python replay_session.py --log logs/demonstrations/log.jsonl --delay 300 \
    --model bc_model.pt --accuracy-out metrics.json
```

During playback the frame, environment state and predicted action confidence are
displayed. Mismatched predictions are highlighted in red. After exiting, a JSON
file is produced containing overall accuracy, a confusion matrix and per-action
statistics. Press **q** to exit the viewer.

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
