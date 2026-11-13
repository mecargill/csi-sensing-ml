"""Shared data preparation and training loop for Widar gesture models."""
from typing import Callable, Optional

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

def normalize(X):
    """NOT how Widar does it... widar does per time instant. 
    but that disregards amplitude variations over time"""
    minimum = X.min(axis=(1, 2, 3), keepdims=True)
    maximum = X.max(axis=(1, 2, 3), keepdims=True)
    return (X - minimum) / (maximum - minimum + 1e-8)

def augment_data(X: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """Apply data augmentation to the input tensor X."""
    #note: in my experiments, this does NOT seem to help. it hurts performance
    noise = torch.randn_like(X) * noise_level
    return (X + noise).clamp(0.0, 1.0)

def train_model(
    model_factory: Callable[[int, int], nn.Module],
    output_path: str,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = 0,
    optimizer: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
    anneal: bool = False,
    augment: float = 0.0,
) -> nn.Module:
    """Load Widar data, train a model, and save its weights."""
    meta = widar_data.load_metadata()
    meta = meta[meta.gesture.isin(COMMON_GESTURES)].reset_index(drop=True)
    print(f"{len(meta)} samples of {len(COMMON_GESTURES)} gestures")

    X, y, classes = widar_data.load_arrays(meta)
    #shape of X is (N, T, H, W)
    X = normalize(X)
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    indices = rng.permutation(len(X))
    n_train = int(0.7 * len(X))
    n_val = int(0.15*len(X))
    train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    X_t = torch.as_tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)

    model = model_factory(X.shape[1], len(classes)).to(device)
    print(f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters")
    loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer(model)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5) if anneal else None

    def accuracy(indices):
        was_training = model.training
        model.eval()
        correct = torch.zeros((), device=device)
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
                inputs = X_t[batch].to(device)
                targets = y_t[batch].to(device)
                correct += (model(inputs).argmax(dim=1) == targets).sum()
        model.train(was_training)
        return correct.item() / len(indices)

    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        total_loss = torch.zeros((), device=device)
        train_correct = torch.zeros((), device=device)
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start:start + batch_size]
            batch_indices = torch.as_tensor(batch, dtype=torch.long)
            batch_t = X_t[batch_indices].to(device)
            targets = y_t[batch_indices].to(device)
            if augment > 0:
                batch_t = augment_data(batch_t, noise_level=augment)
            logits = model(batch_t)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach() * len(batch)
            train_correct += (logits.detach().argmax(dim=1) == targets).sum()
        val_acc = accuracy(val_idx)
        test_acc = accuracy(test_idx)
        train_loss = total_loss.item() / len(train_idx)
        train_acc = train_correct.item() / len(train_idx)

        info = f"epoch {epoch + 1:2d}  loss {train_loss:.3f}  " \
               f"train acc {train_acc:.3f}  val acc {val_acc:.3f}  test acc {test_acc:.3f}"
        
                
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

            info += f"  BEST so far"

        print(info)
        if anneal:
            sched.step()
    if best_model_state is None:
        raise RuntimeError("No best model checkpoint was produced; check epochs and validation data.")
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, output_path)
    print(f"Saved best model (val acc {best_val_acc:.3f}) to {output_path}")
    test_acc = accuracy(test_idx)
    print(f"Best epoch {best_epoch}  val acc {best_val_acc:.3f}  test acc {test_acc:.3f}")
    model.eval()
    return model
