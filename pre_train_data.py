"""Behavior cloning pretraining script."""

import pickle
import argparse
from typing import Sequence

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


def train_bc(demo_file: str, output: str, epochs: int = 5, batch_size: int = 32):
    with open(demo_file, "rb") as f:
        data: Sequence = pickle.load(f)

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

