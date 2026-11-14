"""Shared data preparation and training loop for Widar gesture models."""
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import widar_data as wd


def normalize(X):
    """Normalize each sample across time and space, preserving relative changes over time.

    Widar instead normalizes each time instant independently.
    """
    minimum = X.min(axis=(1, 2, 3), keepdims=True)
    maximum = X.max(axis=(1, 2, 3), keepdims=True)
    return (X - minimum) / (maximum - minimum + 1e-8)


def augment_data(X: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """Apply data augmentation to the input tensor X."""
    # In experiments, this augmentation has reduced performance.
    noise = torch.randn_like(X) * noise_level
    return (X + noise).clamp(0.0, 1.0)

def generate_splits(rng, X, use_validation, lodo_domain, lodo_idx, meta):
    if lodo_domain is None:
        indices = rng.permutation(len(X))
        if use_validation:
            n_train = int(0.7 * len(X))
            n_val = int(0.15 * len(X))
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val:]
        else:
            n_train = int(0.8 * len(X))
            train_idx = indices[:n_train]
            val_idx = np.empty(0, dtype=indices.dtype)
            test_idx = indices[n_train:]
        return train_idx, val_idx, test_idx
    else:
        domain = meta[lodo_domain].to_numpy()
        test_idx = np.flatnonzero(domain == lodo_idx)
        other_idx = rng.permutation(np.flatnonzero(domain != lodo_idx))

        if use_validation:
            n_train = int(0.8 * len(other_idx))
            train_idx = other_idx[:n_train]
            val_idx = other_idx[n_train:]
        else:
            train_idx = other_idx
            val_idx = np.empty(0, dtype=int)

        return train_idx, val_idx, test_idx

#create an easy way to check accuracy
def accuracy(model, X_t, y_t, indices, device, batch_size):
    was_training = model.training
    model.eval()
    correct = torch.zeros((), device=device)

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = torch.as_tensor(
                indices[start : start + batch_size],
                dtype=torch.long,
            )
            inputs = X_t[batch_indices].to(device)
            targets = y_t[batch_indices].to(device)
            correct += (model(inputs).argmax(dim=1) == targets).sum()

    model.train(was_training)
    return correct.item() / len(indices)


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
    use_validation: bool = True,
    lodo_domain = None,
    lodo_idx = None,
    cached_data = None
) -> nn.Module:
    """Train with validation, or use an 80/20 train/test split.

    With validation enabled, save the epoch with the best validation accuracy.
    Without validation, save the final epoch and evaluate test accuracy once.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    
    if cached_data is not None:
        X, y, classes, meta = cached_data
    else:
        #load the metadata (path, date, room, user, gesture, location, orientation, repetition)
        meta = wd.load_metadata()
        meta = meta[meta.gesture.isin(wd.COMMON_GESTURES)].reset_index(drop=True)
        print(f"{len(meta)} samples of {len(wd.COMMON_GESTURES)} gestures")
        X, y, classes = wd.load_arrays(meta)

    assert (lodo_domain is None) == (lodo_idx is None), "lodo_domain and lodo_idx must be specified together."
    assert (lodo_domain is None) or (lodo_domain in meta.columns), "lodo_domain must be a column in the metadata."
    assert (lodo_idx is None) or (lodo_idx in meta[lodo_domain].values), "lodo_idx must be a value in the specified lodo_domain column."
    
    # X has shape (N, T, H, W). Normalize the data to [0, 1]
    X = normalize(X)

    #generate the splits
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = generate_splits(rng, X, use_validation, lodo_domain, lodo_idx, meta)

    #select the device for training
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    X_t = torch.as_tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)

    #create the model, loss, optimizer, and LR scheduler
    model = model_factory(X.shape[1], len(classes)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"{parameter_count:,} parameters")

    loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer(model)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
        if anneal
        else None
    )

    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None

    #MAIN TRAINING LOOP
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        total_loss = torch.zeros((), device=device)
        train_correct = torch.zeros((), device=device)
        #pass once through training data
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            batch_indices = torch.as_tensor(batch, dtype=torch.long)
            batch_inputs = X_t[batch_indices].to(device)
            targets = y_t[batch_indices].to(device)

            if augment > 0:
                batch_inputs = augment_data(batch_inputs, noise_level=augment)

            logits = model(batch_inputs)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach() * len(batch)
            train_correct += (logits.detach().argmax(dim=1) == targets).sum()

        #print info
        train_loss = total_loss.item() / len(train_idx)
        train_acc = train_correct.item() / len(train_idx)
        if use_validation:
            val_acc = accuracy(model, X_t, y_t, val_idx, device, batch_size)
        test_acc = accuracy(model, X_t, y_t, test_idx, device, batch_size)

        info = (
            f"epoch {epoch + 1:2d}  loss {train_loss:.3f}  "
            f"train acc {train_acc:.3f}"
        )

        if use_validation:
            info += f"  val acc {val_acc:.3f}"
            #if validating, keep track of the best val accuracy model parameters
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                info += "  BEST so far"

        info += f"  test acc {test_acc:.3f}"
        print(info)

        #if necessary update LR schedule
        if anneal:
            scheduler.step()

    #If we're validating, load the best val acc model. Else, use final model
    if use_validation:
        if best_model_state is None:
            raise RuntimeError("No best model checkpoint was produced; check epochs and validation data.")
        model.load_state_dict(best_model_state)
        selected_model_state = best_model_state
        print(f"Best epoch {best_epoch}  val acc {best_val_acc:.3f}")
    else:
        selected_model_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        print(f"Using final epoch {epochs}; no validation split was used.")

    #save the weights
    torch.save(selected_model_state, output_path)
    print(f"Saved model weights to {output_path}")

    #final results
    test_acc = accuracy(model, X_t, y_t, test_idx, device, batch_size)
    print(f"Test acc {test_acc:.3f}")
    model.eval()
    return model
