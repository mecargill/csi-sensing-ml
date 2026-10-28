"""Shared data preparation and training loop for Widar gesture models."""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

import widar_data

COMMON_GESTURES = [
    "Push&Pull",
    "Sweep",
    "Clap",
    "Slide",
    "Draw-O(H)",
    "Draw-Zigzag(H)",
]


def train_model(
    model_factory: Callable[[int, int], nn.Module],
    output_path: str,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = 0,
) -> nn.Module:
    """Load Widar data, train a model, and save its weights."""
    meta = widar_data.load_metadata()
    meta = meta[meta.gesture.isin(COMMON_GESTURES)].reset_index(drop=True)
    print(f"{len(meta)} samples of {len(COMMON_GESTURES)} gestures")

    X, y, classes = widar_data.load_arrays(meta)
    X = X.transpose(0, 3, 1, 2)
    X /= X.max() + 1e-8

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    X_t = torch.as_tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    model = model_factory(X.shape[1], len(classes)).to(device)
    print(f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def accuracy(indices):
        model.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(indices), 512):
                batch = indices[start:start + 512]
                correct += (model(X_t[batch]).argmax(dim=1) == y_t[batch]).sum().item()
        model.train()
        return correct / len(indices)

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        total_loss = 0.0
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start:start + batch_size]
            loss = loss_fn(model(X_t[batch]), y_t[batch])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"epoch {epoch + 1:2d}  loss {total_loss / len(train_idx):.3f}  "
              f"train acc {accuracy(train_idx):.3f}  test acc {accuracy(test_idx):.3f}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved model weights to {output_path}")
    return model
