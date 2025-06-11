"""Behavior cloning pretraining script."""

import pickle
import argparse
import json
from typing import Sequence, List, Tuple, Dict

from src.env import EveEnv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class BCModel(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
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


def _load_jsonl(path: str, env: EveEnv) -> Sequence[Tuple[List[float], int]]:
    mapping = _label_mapping(env)
    data = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            label = entry.get('action')
            if label in mapping:
                action = mapping[label]
            elif label and '_' in label:
                try:
                    action = int(label.split('_')[-1])
                except ValueError:
                    continue
            else:
                continue
            obs = entry.get('state', {}).get('obs')
            if obs is not None:
                data.append((obs, action))
    return data


def _load_data(demo_file: str) -> Sequence[Tuple[List[float], int]]:
    env = EveEnv()
    if demo_file.endswith('.jsonl'):
        return _load_jsonl(demo_file, env)
    with open(demo_file, 'rb') as f:
        return pickle.load(f)


def train_bc(demo_file: str, output: str, epochs: int = 5, batch_size: int = 32):
    data: Sequence = _load_data(demo_file)

    obs = torch.tensor([d[0] for d in data], dtype=torch.float32)
    actions = torch.tensor([d[1] for d in data], dtype=torch.long)

    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCModel(obs.shape[1], actions.max().item() + 1)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for b_obs, b_act in loader:
            logits = model(b_obs)
            loss = loss_fn(logits, b_act)
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

