"""Run every available LODO split for the configured LSTM and temporal ResNet."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

import widar_data as wd
import widar_training as wt
from widar_models import LSTM, TemporalResNet
import sys
from datetime import datetime, timedelta
import time

EPOCHS = 100
BATCH_SIZE = 64
SEED = 0
DOMAINS = ("room", "user", "location", "orientation")
OUTPUT_DIR = Path("lodo")
MODEL_DIR = OUTPUT_DIR / Path("models")
now = datetime.now()
LOG_FPATH = OUTPUT_DIR / Path("logs") / f"log_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_file = open(LOG_FPATH, "w", buffering=1)

original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)


def load_cached_data():
    meta = pd.read_csv(wd.PRE_PARSED_DATA_PATH / "meta.csv")
    X = np.load(wd.PRE_PARSED_DATA_PATH / "X.npy", mmap_mode="r")
    y = np.load(wd.PRE_PARSED_DATA_PATH / "y.npy")
    with open(wd.PRE_PARSED_DATA_PATH / "classes.txt", "r", encoding="utf-8") as file:
        classes = file.read().splitlines()

    total_runs = 0
    for domain in DOMAINS:
        unique = meta[domain].unique()
        print(f"unique {domain}s: {unique}")
        total_runs += len(unique)
    print(f"total runs: {total_runs}")
    return (X, y, classes, meta)

prgrm_start = time.perf_counter()
def run_lodo_runs(model_name, model_factory, cached_data, **settings):
    meta = cached_data[3]
    
    print("\n\n--------------------------------------------------------------------------")
    print(f"Running in-domain test for {model_name} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    print(f"Time elapsed: {timedelta(seconds=round(time.perf_counter() - prgrm_start))}")
    print("--------------------------------------------------------------------------")
    output_path = MODEL_DIR / f"{model_name}_in_domain.pt"
    wt.train_model(
                    model_factory,
                    str(output_path),
                    cached_data=cached_data,
                    **settings,
                )
    for domain in DOMAINS:
        for lodo_idx in pd.unique(meta[domain].dropna()):
            output_path = MODEL_DIR / f"{model_name}_{domain}_{lodo_idx}.pt"
            print("\n\n--------------------------------------------------------------------------")
            print(f"Running {model_name}: {domain}={lodo_idx} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
            print(f"Time elapsed: {timedelta(seconds=round(time.perf_counter() - prgrm_start))}")
            print("--------------------------------------------------------------------------")
            wt.train_model(
                model_factory,
                str(output_path),
                lodo_domain=domain,
                lodo_idx=lodo_idx,
                cached_data=cached_data,
                **settings,
            )


def export_test_accuracies(log_path):
    results = []
    current_run = None
    in_domain_pattern = re.compile(r"Running in-domain test for (\S+)")
    lodo_pattern = re.compile(r"Running (\S+): (\w+)=(\S+)")
    accuracy_pattern = re.compile(r"Test acc ([0-9]*\.?[0-9]+)")

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = in_domain_pattern.match(line)
        if match:
            current_run = {
                "model": match.group(1),
                "split": "in_domain",
                "domain": "",
                "held_out": "",
            }
            continue

        match = lodo_pattern.match(line)
        if match:
            current_run = {
                "model": match.group(1),
                "split": "lodo",
                "domain": match.group(2),
                "held_out": match.group(3),
            }
            continue

        match = accuracy_pattern.match(line)
        if match:
            if current_run is None:
                raise ValueError(f"Test accuracy in {log_path} has no associated run header.")
            results.append({**current_run, "test_accuracy": float(match.group(1))})
            current_run = None

    if current_run is not None:
        raise ValueError(f"Run in {log_path} has no final test accuracy.")
    if not results:
        raise ValueError(f"No test accuracies found in {log_path}.")

    output_path = log_path.with_name(f"{log_path.stem}_test_accuracies.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cached_data()
    """
    run_lodo_runs(
        "lstm",
        LSTM,
        data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        anneal=True,
        normalization="sample_minmax",
        augment=False,
        all_gestures=False,
        use_validation=False,
    )"""
    
    
    run_lodo_runs(
        "temporal_resnet_v2",
        TemporalResNet,
        data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        optimizer=lambda model: torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.05
        ),
        one_cycle=True,
        normalization="fit_max_sqrt",
        augment=True,
        all_gestures=True,
        use_validation=False,
    )

    sys.stdout.flush()
    sys.stdout = original_stdout
    log_file.close()
    accuracy_csv = export_test_accuracies(LOG_FPATH)
    print(f"Test accuracies saved to {accuracy_csv}")
