"""Run every available LODO split for the configured LSTM and temporal ResNet."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import widar_data as wd
import widar_training as wt
from widar_models import LSTM, TemporalResNet


EPOCHS = 100
BATCH_SIZE = 64
SEED = 0
DOMAINS = ("room", "user", "location", "orientation")
OUTPUT_DIR = Path("lodo")


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

def run_lodo_runs(model_name, model_factory, cached_data, **settings):
    meta = cached_data[3]
    print("\n\n--------------------------------------------------------------------------")
    print(f"Running in-domain test for {model_name}")
    print("--------------------------------------------------------------------------")
    output_path = OUTPUT_DIR / f"{model_name}_in_domain.pt"
    wt.train_model(
                    model_factory,
                    str(output_path),
                    cached_data=cached_data,
                    **settings,
                )
    for domain in DOMAINS:
        for lodo_idx in pd.unique(meta[domain].dropna()):
            output_path = OUTPUT_DIR / f"{model_name}_{domain}_{lodo_idx}.pt"
            print("\n\n--------------------------------------------------------------------------")
            print(f"Running {model_name}: {domain}={lodo_idx}")
            print("--------------------------------------------------------------------------")
            wt.train_model(
                model_factory,
                str(output_path),
                lodo_domain=domain,
                lodo_idx=lodo_idx,
                cached_data=cached_data,
                **settings,
            )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cached_data()
    
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
    )
    

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
