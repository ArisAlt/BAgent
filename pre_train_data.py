"""Behavior cloning pretraining script."""

# version: 0.2.0 | path: pre_train_data.py

import pickle
import argparse
import json
from typing import Sequence, List, Tuple, Dict

from PIL import Image
import numpy as np

from src.env import EveEnv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class BCModel(nn.Module):
    """Simple hybrid CNN + MLP model for behavior cloning."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out = 16 * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out + obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, inputs):
        img, state = inputs
        feats = self.cnn(img)
        combined = torch.cat([feats, state], dim=1)
        return self.mlp(combined)


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

    # Load images and states
    images = []
    states = []
    actions_idx = []
    for frame, obs, act in data:
        img = Image.open(frame).convert("RGB").resize((64, 64))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        images.append(img_tensor)
        states.append(torch.tensor(obs, dtype=torch.float32))
        actions_idx.append(act)

    images = torch.stack(images)
    states = torch.stack(states)
    actions_idx = torch.tensor(actions_idx, dtype=torch.long)

    # Normalize observations
    mean = states.mean(0)
    std = states.std(0) + 1e-8
    states = (states - mean) / std

    n_actions = actions_idx.max().item() + 1
    actions = F.one_hot(actions_idx, num_classes=n_actions).float()

    dataset = TensorDataset(images, states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCModel(states.shape[1], n_actions)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for b_img, b_state, b_act in loader:
            logits = model([b_img, b_state])
            target = torch.argmax(b_act, dim=1)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), output)
    print(f"Saved BC model to {output}")


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

