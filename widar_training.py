"""Shared data preparation and training loop for Widar gesture models."""
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import widar_data as wd


def filter_preparsed_data(X, y, classes, meta, all_gestures):
    if len(X) != len(y) or len(y) != len(meta):
        raise ValueError("Pre-parsed X, y, and metadata must contain the same number of samples.")

    y = np.asarray(y)
    if y.ndim != 1 or (y.size and (y.min() < 0 or y.max() >= len(classes))):
        raise ValueError("Pre-parsed labels do not match the saved classes.")
    if not np.array_equal(np.asarray(classes)[y], meta.gesture.to_numpy()):
        raise ValueError("Pre-parsed labels and metadata gestures are not aligned.")

    keep = np.ones(len(meta), dtype=bool)
    if not all_gestures:
        keep &= meta.gesture.isin(wd.COMMON_GESTURES).to_numpy()
    suspect = (meta.date.astype(str) == "20181109") & (meta.gesture == "Slide")
    keep &= ~suspect.to_numpy()

    indices = np.flatnonzero(keep)
    meta = meta.iloc[indices].reset_index(drop=True)
    X = X[indices]
    classes = sorted(meta.gesture.unique())
    if not classes:
        raise ValueError("No samples remain after filtering the pre-parsed data.")
    class_to_index = {gesture: index for index, gesture in enumerate(classes)}
    y = meta.gesture.map(class_to_index).to_numpy(dtype=np.int64)
    return X, y, classes, meta


def normalize(X, mode="sample_minmax", fit_indices=None):
    """Normalize Widar samples using the selected training recipe."""
    if mode == "sample_minmax":
        minimum = X.min(axis=(1, 2, 3), keepdims=True)
        maximum = X.max(axis=(1, 2, 3), keepdims=True)
        return (X - minimum) / (maximum - minimum + 1e-8)
    if mode == "global_max":
        return X / (X.max() + 1e-8)
    if mode == "fit_max_sqrt":
        if fit_indices is None or len(fit_indices) == 0:
            raise ValueError("fit_indices must contain training samples for fit_max_sqrt normalization.")
        scale = X[fit_indices].max()
        if scale <= 0:
            raise ValueError("fit_max_sqrt normalization requires a positive training-data maximum.")
        return np.sqrt(np.clip(X, 0, None) / scale)
    raise ValueError(f"Unknown normalization mode: {mode}")



def random_shift(x, dim, lo, hi):
    """Slide every sample along one axis by its own random amount (lo..hi steps); emptied slots become 0.

    Each sample gets a different shift on purpose. Shifting a whole batch the same way
    makes BatchNorm's running statistics jump around, and test accuracy jumps with them.
    """
    B, n = len(x), x.shape[dim]
    k = torch.randint(lo, hi + 1, (B, 1), device=x.device)
    src = torch.arange(n, device=x.device) - k                # (B, n): which old position lands in each slot
    shape = [B, 1, 1, 1]
    shape[dim] = n                                            # so src lines up with the axis being shifted
    inside = ((src >= 0) & (src < n)).view(shape)             # slots whose source fell off the edge stay empty
    return x.gather(dim, src.clamp(0, n - 1).view(shape).expand_as(x)) * inside


def augment_data(x):
    """Return a randomly shifted copy of a training batch x: (B, 30, 20, 20).

    The label does not change if the gesture starts a little later or the velocity
    image moves by one pixel, so the model sees a slightly different version of
    every sample each epoch and cannot simply memorize them.
    """
    x = random_shift(x, 1, 0, 6)     # time: start 0..6 frames later (clips are ~16 frames, zero-padded to 30)
    x = random_shift(x, 2, -1, 1)    # velocity image: one pixel either way, on each axis
    x = random_shift(x, 3, -1, 1)
    return x

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
    augment: bool = False,
    use_validation: bool = True,
    one_cycle: bool = False,
    normalization: str = "sample_minmax",
    all_gestures: bool = False,
    lodo_domain = None,
    lodo_idx = None,
    cached_data = None,
    use_pre_parsed_data = False
) -> nn.Module:
    """Train with random or LODO splits and an optional validation subset.

    With validation enabled, save the epoch with the best validation accuracy.
    Without validation, train on all non-test samples and save the final epoch.
    LODO test indices contain the entire held-out domain.
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if anneal and one_cycle:
        raise ValueError("anneal and one_cycle cannot both be enabled.")
    if use_pre_parsed_data and (cached_data is not None):
        print("WARNING: preparsed data was true... but cached data was passed. cached data was used.")

    if cached_data is not None:
        X, y, classes, meta = cached_data
    else:
        if not use_pre_parsed_data:
            #load the metadata (path, date, room, user, gesture, location, orientation, repetition)
            meta = wd.load_metadata()
            print(f"{len(meta)} samples of {meta.gesture.nunique()} gestures")
            X, y, classes = wd.load_arrays(meta)
        else:
            meta = pd.read_csv(wd.PRE_PARSED_DATA_PATH / "meta.csv")
            X = np.load(wd.PRE_PARSED_DATA_PATH / "X.npy")
            y = np.load(wd.PRE_PARSED_DATA_PATH / "y.npy")
            with open(wd.PRE_PARSED_DATA_PATH / "classes.txt", "r", encoding="utf-8") as f:
                classes = f.read().splitlines()
    X, y, classes, meta = filter_preparsed_data(
        X, y, classes, meta, all_gestures
    )

    assert (lodo_domain is None) == (lodo_idx is None), "lodo_domain and lodo_idx must be specified together."
    assert (lodo_domain is None) or (lodo_domain in meta.columns), "lodo_domain must be a column in the metadata."
    assert (lodo_idx is None) or (lodo_idx in meta[lodo_domain].values), "lodo_idx must be a value in the specified lodo_domain column."
    
    #generate the splits
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = generate_splits(rng, X, use_validation, lodo_domain, lodo_idx, meta)
    # Fit-set normalization uses only training samples, never validation or LODO test data.
    X = normalize(X, mode=normalization, fit_indices=train_idx)

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

    if one_cycle:
        steps_per_epoch = (len(train_idx) + batch_size - 1) // batch_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=epochs * steps_per_epoch,
        )
    elif anneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
    else:
        scheduler = None

    model_name = getattr(model_factory, "__name__", type(model_factory).__name__)
    print("Training settings:")
    print(f"  model: {model_name}")
    print(f"  output: {output_path}")
    print(f"  epochs: {epochs}; batch size: {batch_size}; seed: {seed}")
    print(f"  data source: {'cached_data' if cached_data is not None else 'pre-parsed' if use_pre_parsed_data else 'raw BVP'}")
    print(f"  classes: {'all gestures' if all_gestures else 'common gestures'} ({len(classes)} classes)")
    print(f"  normalization: {normalization}")
    print(f"  augmentation: {'random shifts' if augment else 'off'}")
    print(f"  validation: {'on' if use_validation else 'off'}")
    split_description = (
        f"Held out {lodo_domain} number {lodo_idx}" if lodo_domain is not None else "random split"
    )
    print(
        f"  split: {split_description}; "
        f"train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}"
    )
    print(f"  loss: {loss_fn.__class__.__name__}")
    print(f"  optimizer: {optimizer.__class__.__name__}")
    for group_index, group in enumerate(optimizer.param_groups, start=1):
        optimizer_options = (
            f"{name}={group[name]}"
            for name in ("lr", "weight_decay", "momentum", "alpha", "betas", "eps")
            if name in group
        )
        print(f"    group {group_index}: {', '.join(optimizer_options)}")
    if one_cycle:
        print(
            f"  scheduler: OneCycleLR (max_lr=1e-3, "
            f"total_steps={epochs * steps_per_epoch}, per batch)"
        )
    elif anneal:
        print(f"  scheduler: CosineAnnealingLR (T_max={epochs}, eta_min=1e-5, per epoch)")
    else:
        print("  scheduler: off")

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

            if augment:
                batch_inputs = augment_data(batch_inputs)

            logits = model(batch_inputs)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if one_cycle:
                scheduler.step()
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
