"""Loader for Widar 3.0 BVP features.

Files: BVP/<date>-VS/[6-link/]user<id>/user<id>-<gesture>-<location>-<orientation>-<rep>-....mat
Each holds "velocity_spectrum_ro", a 20x20xT array (T varies, 4..30 at 10 Hz).
Gesture ids are numbered per collection date (see README.pdf), so they are mapped to names here.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import glob
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

ROOT = Path(__file__).resolve().parent / "BVP"

ROOM = {"20181109": 1, "20181112": 1, "20181115": 1, "20181117": 2, "20181118": 2, "20181121": 1,
        "20181127": 2, "20181128": 2, "20181130": 1, "20181204": 2, "20181205": 2, "20181208": 2,
        "20181209": 2, "20181211": 3}

_A = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-Zigzag(V)", "Draw-N(V)"]
_B = ["Push&Pull", "Sweep", "Clap", "Draw-O(V)", "Draw-Zigzag(V)", "Draw-N(V)"]
_C = ["Slide", "Draw-O(H)", "Draw-Zigzag(H)", "Draw-N(H)", "Draw-Triangle(H)", "Draw-Rectangle(H)"]
_D = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(H)", "Draw-Zigzag(H)", "Draw-N(H)", "Draw-Triangle(H)", "Draw-Rectangle(H)"]
_DIGITS = [f"Draw-{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
# date -> list indexed by gesture id - 1, or {user: list} when it differs per user
GESTURES = {"20181109": _A, "20181112": _DIGITS, "20181115": _B, "20181117": _B, "20181118": _B,
            "20181121": _C, "20181127": _C, "20181128": _D[:3] + _D[4:7], "20181130": _D, "20181204": _D,
            "20181205": {2: _C[1:], 3: _C},
            "20181208": {2: _D[:4], 3: _D[:3]},
            "20181209": {2: _D[:1], 6: _D[:6]},
            "20181211": _D[:6]}


def load_metadata():
    """One row per sample: path, date, room, user, gesture (name), location, orientation, rep."""
    started = time.perf_counter()
    print("Indexing Widar metadata...", flush=True)
    rows = []
    for p in glob.glob(str(ROOT / "*" / "**" / "*.mat"), recursive=True):
        if Path(p).stat().st_size == 0:  # one file in the release is empty (packet loss)
            continue
        date = Path(p).relative_to(ROOT).parts[0][:8]
        user, g, loc, ori, rep = map(int, re.match(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", Path(p).name).groups())
        names = GESTURES[date]
        names = names[user] if isinstance(names, dict) else names
        rows.append(dict(path=p, date=date, room=ROOM[date], user=user, gesture=names[g - 1],
                         location=loc, orientation=ori, rep=rep))
    print(f"Indexed {len(rows):,} Widar samples in {time.perf_counter() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def load_sample(path, T=30):
    """Return (20, 20, T) float32, zero-padded or truncated along time."""
    x = scipy.io.loadmat(path)["velocity_spectrum_ro"].astype(np.float32)
    x = x.reshape(20, 20, -1) if x.size else np.zeros((20, 20, 0), np.float32)  # 1-frame clips are stored 2-D
    out = np.zeros((20, 20, T), np.float32)
    n = min(T, x.shape[2])
    out[:, :, :n] = x[:, :, :n]
    return out


def load_arrays(meta, T=30):
    """X: (N, 20, 20, T). y: gesture class indices. classes: index -> name."""
    total = len(meta)
    X = np.empty((total, 20, 20, T), dtype=np.float32)
    update_every = max(1, (total + 19) // 20)
    batch_size = 256
    workers = min(4, max(1, total))
    started = time.perf_counter()
    print(f"Loading {total:,} Widar samples with {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch_start in range(0, total, batch_size):
            paths = meta.path.iloc[batch_start:batch_start + batch_size]
            for offset, sample in enumerate(pool.map(partial(load_sample, T=T), paths)):
                X[batch_start + offset] = sample
                completed = batch_start + offset + 1
                if completed == 1 or completed % update_every == 0 or completed == total:
                    elapsed = time.perf_counter() - started
                    if completed < update_every:
                        timing = f"{elapsed:.1f}s elapsed, estimating time remaining"
                    else:
                        remaining = elapsed * (total - completed) / completed
                        timing = f"{elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                    print(f"Loaded {completed:,}/{total:,} samples "
                          f"({completed / total:.0%}) - {timing}", flush=True)
    classes = sorted(meta.gesture.unique())
    y = meta.gesture.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return X, y, classes
