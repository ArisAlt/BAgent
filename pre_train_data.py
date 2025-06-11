"""Behavior cloning pretraining script."""

# version: 0.3.0 | path: pre_train_data.py

import pickle
import argparse
import json
from typing import Sequence, List, Tuple, Dict

import numpy as np

from src.env import EveEnv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BCModel(nn.Module):
    """Simple MLP model for behavior cloning."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _label_mapping(env: EveEnv) -> Dict[str, int]:
    mapping = {}
    for idx, (typ, target) in enumerate(env.actions):
        if typ == 'click':
            label = f'click_{target}'
        elif typ == 'keypress':
            label = f'keypress_{target}'
        else:
            label = 'sleep'
        mapping[label] = idx
    return mapping


def _load_jsonl(path: str, env: EveEnv) -> Sequence[Tuple[str, List[float], int]]:
    """Load data from a jsonl demonstration file.

    Returns a list of ``(frame_path, obs, action_idx)`` tuples.
    """
    mapping = _label_mapping(env)
    data = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            label = entry.get("action")
            if label in mapping:
                action = mapping[label]
            elif label and "_" in label:
                try:
                    action = int(label.split("_")[-1])
                except ValueError:
                    continue
            else:
                continue
            obs = entry.get("state", {}).get("obs")
            frame = entry.get("frame")
            if obs is not None and frame:
                data.append((frame, obs, action))
    return data


def _load_data(demo_file: str) -> Sequence[Tuple[str, List[float], int]]:
    """Load demonstration data from jsonl or pickle."""
    env = EveEnv()
    if demo_file.endswith(".jsonl"):
        return _load_jsonl(demo_file, env)
    with open(demo_file, "rb") as f:
        return pickle.load(f)


def train_bc(demo_file: str, output: str, epochs: int = 5, batch_size: int = 32):
    """Train a simple behavior cloning model from demonstration data."""
    data = _load_data(demo_file)

    # Load states only
    states = []
    actions_idx = []
    for _, obs, act in data:
        states.append(obs)
        actions_idx.append(act)

    states = np.array(states, dtype=np.float32)
    actions_idx = np.array(actions_idx, dtype=np.int64)

    # Standardize observations
    scaler = StandardScaler()
    states = scaler.fit_transform(states)

    # Train/validation split
    indices = np.arange(len(states))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    state_train = torch.tensor(states[train_idx])
    y_train = torch.tensor(actions_idx[train_idx])
    state_val = torch.tensor(states[val_idx])
    y_val = torch.tensor(actions_idx[val_idx])

    n_actions = actions_idx.max() + 1

    train_ds = TensorDataset(state_train, y_train)
    val_ds = TensorDataset(state_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = BCModel(states.shape[1], n_actions)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for b_state, b_act in train_loader:
            logits = model(b_state)
            loss = loss_fn(logits, b_act)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_state, v_act in val_loader:
                v_logits = model(v_state)
                val_loss += loss_fn(v_logits, v_act).item() * v_state.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f}")

    torch.save({'model_state': model.state_dict(), 'scaler': scaler}, output)
    print(f"Saved BC model and scaler to {output}")


def main():
    parser = argparse.ArgumentParser(description="Train behavior cloning model")
    parser.add_argument("--demos", type=str, default="demo_buffer.pkl")
    parser.add_argument("--out", type=str, default="bc_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    train_bc(args.demos, args.out, epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()

