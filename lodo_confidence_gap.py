"""Measure train-versus-test confidence for available LODO checkpoints."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import widar_data as wd
import widar_training as wt
from widar_models import LSTM, TemporalResNet


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "lodo" / "models"
LOG_DIR = ROOT / "lodo" / "logs"
DOMAINS = ("room", "user", "location", "orientation")
SEED = 0
BATCH_SIZE = 64

MODEL_CONFIGS = (
    {
        "name": "lstm",
        "display_name": "LSTM",
        "factory": LSTM,
        "all_gestures": False,
        "normalization": "sample_minmax",
    },
    {
        "name": "temporal_resnet_v2",
        "display_name": "Temporal ResNet v2",
        "factory": TemporalResNet,
        "all_gestures": True,
        "normalization": "fit_max_sqrt",
    },
)


def load_cached_data():
    data_dir = wd.PRE_PARSED_DATA_PATH
    meta = pd.read_csv(data_dir / "meta.csv")
    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    with (data_dir / "classes.txt").open(encoding="utf-8") as classes_file:
        classes = classes_file.read().splitlines()
    return X, y, classes, meta


def available_splits(meta, model_name):
    candidates = [(None, None, MODEL_DIR / f"{model_name}_in_domain.pt")]
    for domain in DOMAINS:
        for held_out in pd.unique(meta[domain].dropna()):
            if isinstance(held_out, (int, np.integer)) or float(held_out).is_integer():
                identifier = int(held_out)
            else:
                identifier = held_out
            candidates.append(
                (
                    domain,
                    identifier,
                    MODEL_DIR / f"{model_name}_{domain}_{identifier}.pt",
                )
            )
    existing = [candidate for candidate in candidates if candidate[2].is_file()]
    missing = [candidate[2].name for candidate in candidates if not candidate[2].is_file()]
    return existing, missing


def prepare_model_data(config, raw_data):
    X, y, original_classes, original_meta = raw_data
    X, y, classes, meta = wt.filter_preparsed_data(
        X,
        y,
        original_classes,
        original_meta,
        all_gestures=config["all_gestures"],
    )
    missing_common = set(wd.COMMON_GESTURES).difference(classes)
    if missing_common:
        raise ValueError(f"Common gestures are missing from the filtered classes: {sorted(missing_common)}")
    return X, y, classes, meta


def checkpoint_output_count(config, state_dict):
    weight_key = "out.weight" if config["name"] == "lstm" else "head.3.weight"
    if weight_key not in state_dict or state_dict[weight_key].ndim != 2:
        raise ValueError(f"Cannot determine output class count from checkpoint key {weight_key!r}")
    return int(state_dict[weight_key].shape[0])


def validate_checkpoint_output_count(config, checkpoint, output_count, expected_count):
    if output_count == expected_count:
        return
    if config["name"] == "lstm" and output_count == 22:
        raise ValueError(
            f"{checkpoint} has 22 LSTM outputs. LSTM checkpoints must use "
            f"{expected_count} common-gesture outputs; refusing to run this one-off checkpoint."
        )
    raise ValueError(
        f"{checkpoint} has {output_count} outputs, but "
        f"{config['display_name']} checkpoints must have {expected_count}."
    )


def expected_output_count(meta, all_gestures):
    suspect = (meta.date.astype(str) == "20181109") & (meta.gesture == "Slide")
    valid = ~suspect.to_numpy()
    if not all_gestures:
        valid &= meta.gesture.isin(wd.COMMON_GESTURES).to_numpy()
    return int(meta.loc[valid, "gesture"].nunique())


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_max_common_probability(model, X_t, indices, common_indices, device):
    if len(indices) == 0:
        raise ValueError("Cannot compute confidence for an empty split.")

    confidence_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = torch.as_tensor(
                indices[start : start + BATCH_SIZE],
                dtype=torch.long,
            )
            inputs = X_t[batch_indices].to(device)
            logits = model(inputs)
            common_logits = logits.index_select(1, common_indices)
            max_probabilities = torch.softmax(common_logits, dim=1).amax(dim=1)
            confidence_sum += max_probabilities.sum().item()
            sample_count += len(batch_indices)

    return confidence_sum / sample_count


def evaluate_checkpoint(
    config,
    checkpoint,
    state_dict,
    domain,
    held_out,
    prepared_data,
    device,
):
    X, y, classes, meta = prepared_data
    rng = np.random.default_rng(SEED)
    train_idx, _, test_idx = wt.generate_splits(
        rng,
        X,
        use_validation=False,
        lodo_domain=domain,
        lodo_idx=held_out,
        meta=meta,
    )
    X = wt.normalize(X, mode=config["normalization"], fit_indices=train_idx)
    X_t = torch.as_tensor(X)

    common_indices = torch.tensor(
        [classes.index(gesture) for gesture in wd.COMMON_GESTURES],
        dtype=torch.long,
        device=device,
    )
    model = config["factory"](X.shape[1], len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    train_confidence = mean_max_common_probability(
        model, X_t, train_idx, common_indices, device
    )
    test_confidence = mean_max_common_probability(
        model, X_t, test_idx, common_indices, device
    )
    train_common_count = int(
        meta.gesture.iloc[train_idx].isin(wd.COMMON_GESTURES).sum()
    )
    return {
        "model": config["name"],
        "checkpoint": checkpoint.name,
        "split": "in_domain" if domain is None else "lodo",
        "domain": "" if domain is None else domain,
        "held_out": "" if held_out is None else held_out,
        "model_class_count": len(classes),
        "confidence_class_count": len(common_indices),
        "train_samples": len(train_idx),
        "train_common_samples": train_common_count,
        "test_samples": len(test_idx),
        "mean_max_softmax_train": train_confidence,
        "mean_max_softmax_test": test_confidence,
        "confidence_gap_train_minus_test": train_confidence - test_confidence,
    }


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_cached_data()
    raw_meta = raw_data[3]
    device = select_device()
    results = []
    skipped = []

    for config in MODEL_CONFIGS:
        checkpoints, missing = available_splits(raw_meta, config["name"])
        skipped.extend(f"{config['name']}/{name}" for name in missing)
        if not checkpoints:
            print(f"No checkpoints found for {config['display_name']}; skipping this model.")
            continue

        expected_count = expected_output_count(raw_meta, config["all_gestures"])
        compatible_checkpoints = []
        for domain, held_out, checkpoint in checkpoints:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            output_count = checkpoint_output_count(config, state_dict)
            validate_checkpoint_output_count(
                config, checkpoint, output_count, expected_count
            )
            compatible_checkpoints.append((domain, held_out, checkpoint))
            del state_dict

        print(
            f"Evaluating {len(compatible_checkpoints)} {config['display_name']} "
            f"checkpoint(s) on {device}."
        )
        prepared_data = prepare_model_data(config, raw_data)
        for domain, held_out, checkpoint in compatible_checkpoints:
            print(f"  Evaluating {checkpoint.name}")
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            results.append(
                evaluate_checkpoint(
                    config,
                    checkpoint,
                    state_dict,
                    domain,
                    held_out,
                    prepared_data,
                    device,
                )
            )
            del state_dict
        del prepared_data

    if skipped:
        print(f"Skipped {len(skipped)} missing checkpoint(s).")
    if not results:
        raise RuntimeError(
            f"No compatible checkpoints were available to evaluate in {MODEL_DIR}."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_path = LOG_DIR / f"confidence_gaps_{timestamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Confidence results saved to {output_path}")


if __name__ == "__main__":
    main()
