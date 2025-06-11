# BAgent

A toolkit for automating EVE Online interactions. The project includes a Gym environment, UI automation modules, and utilities for OCR and computer vision.

## Installation

```bash
pip install -r requirements.txt
```

Run tests with:

```bash
pytest -q
```

## Behavior Cloning Pretraining

1. Record demonstrations:

```bash
python data_recorder.py --manual False
```

2. Train the BC model:

```bash
python pre_train_data.py --demos demo_buffer.pkl --out bc_model.pt
```

3. Fine-tune with PPO (optional `--bc_model`):

```bash
python run_start.py --train --bc_model bc_model.pt --timesteps 50000
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
