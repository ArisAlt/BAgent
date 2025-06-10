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
