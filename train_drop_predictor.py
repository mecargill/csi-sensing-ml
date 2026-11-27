"""
Can a SMALL model predict how much accuracy the gesture model loses in a new room,
without being told the right answers there?

Nothing big is retrained. This script loads the four saved v2 checkpoints:

  gesture_temporal_resnet_v2.pt      trained in every room            (no room shift)
  gesture_cross_room{1,2,3}_v2.pt    trained with one room held out   (real room shift)

HOW IT WORKS
  1. FEATURES       Run a big model on a clip and stop just before its last layer. That
                    gives 512 numbers: the model's feature representation of the clip.
  2. MEASUREMENTS   Every big model has its own private 512 numbers, so they cannot be
                    compared between models directly. Instead each clip is described by
                    WHERE its features sit compared with that model's training clips:
                      d1    distance to the nearest gesture centre
                      d2    distance to the second-nearest gesture centre
                      norm  length of the feature vector
                    Each is written relative to the same model's held-out same-room
                    clips (0 = typical for a room it knows, +2 = two std devs above).
  3. SMALL MODEL    3 inputs -> 16 -> 1 output, 81 parameters. It answers one question
                    per clip: "how likely is the big model wrong on this clip?"
  4. DROP           predicted drop = average risk on the new clips
                                   - average risk on same-room clips
                    No labels from the new clips are used.

THE TEST
  To predict the drop in room R, the small model learns only from the OTHER big models,
  and from clips recorded in the OTHER rooms. It never sees a clip from room R, and it
  never sees a feature from the big model it is tested on.

  python3 train_drop_predictor.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import widar_data

torch.manual_seed(0)

# ----------------------------------------------------------------------------- 1. DATA (same as the v2 scripts)
COMMON = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(H)", "Draw-Zigzag(H)"]

meta = widar_data.load_metadata()
suspect = (meta.date == "20181109") & (meta.gesture == "Slide")
meta = meta[~suspect].reset_index(drop=True)             # drop the suspect "Slide" session

X, y, classes = widar_data.load_arrays(meta)
X = X.transpose(0, 3, 1, 2)                              # (N, 30, 20, 20) = (sample, time, vx, vy)
X = np.sqrt(np.clip(X, 0, None))                         # the square root from v2; each model's scale is applied later
clip_max = X.reshape(len(X), -1).max(axis=1)             # largest value in each clip, used to rebuild each model's scale

common_cls = [classes.index(g) for g in COMMON]
is_common = np.isin(y, common_cls)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# The two functions below rebuild the exact splits of the training scripts (same seed, same
# order), so "held-out" here means clips the saved model really never trained on.
def split_every_room():
    """Split used by train_widar_temporal_resnet_v2.py: 80 / 20 at random, every room."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(np.flatnonzero(is_common))
    n_train = int(0.8 * len(idx))
    no_new_room = np.array([], dtype=int)                # this model has no unseen room
    return idx[:n_train], idx[n_train:], no_new_room, clip_max.max()


def split_hold_out(room):
    """Split used by train_widar_cross_room_v2.py: one room held out."""
    in_test_room = (meta.room == room).to_numpy()
    rng = np.random.default_rng(0)
    other_rooms = rng.permutation(np.flatnonzero(is_common & ~in_test_room))
    n_train = int(0.9 * len(other_rooms))
    new_room_idx = np.flatnonzero(is_common & in_test_room)
    trained_on = ~in_test_room                           # that script scaled by its training rooms only
    trained_on[other_rooms[n_train:]] = False
    return other_rooms[:n_train], other_rooms[n_train:], new_room_idx, clip_max[trained_on].max()


# ----------------------------------------------------------------------------- 2. BIG MODEL (unchanged, plus a way to read its features)
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class TemporalResNet(nn.Module):
    def __init__(self, n_classes, width=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(400, width, 3, padding=1, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        self.layer1 = ResidualBlock(width, width)
        self.layer2 = ResidualBlock(width, 2 * width, stride=2)
        self.layer3 = ResidualBlock(2 * width, 4 * width, stride=2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * width, n_classes))

    def features(self, x):
        """NEW: stop before the last layer. Returns 512 numbers per clip."""
        x = x.flatten(2).transpose(1, 2)             # (N, 30, 20, 20) -> (N, 400, 30)
        return self.head[:2](self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def forward(self, x):
        return self.head[2:](self.features(x))


big = TemporalResNet(n_classes=len(classes)).to(device)


def run_big(indices, scale):
    """Features (512 per clip) and the 6 gesture scores for the clips in `indices`."""
    feats, scores = [], []
    with torch.no_grad():
        for i in range(0, len(indices), 512):
            f = big.features(torch.tensor(X[indices[i:i + 512]] / scale, device=device))
            feats.append(f.cpu())
            scores.append(big.head[3](f)[:, common_cls].cpu())     # last layer; only the 6 common gestures compete
    return torch.cat(feats).numpy(), torch.cat(scores)


# ----------------------------------------------------------------------------- 3. MEASURE EVERY HELD-OUT CLIP
INPUTS = ["d1", "d2", "norm"]                            # what the small model gets to see


def measure(name, checkpoint, train_idx, same_idx, new_idx, scale):
    """One row per held-out clip of one big model: the 3 measurements, and was the big model wrong?"""
    big.load_state_dict(torch.load(checkpoint, map_location=device))
    big.eval()

    # Where do the TRAINING clips sit? One centre per gesture.
    f_train, s_train = run_big(train_idx, scale)
    mean, std = f_train.mean(axis=0), f_train.std(axis=0) + 1e-6
    f_train = (f_train - mean) / std
    centres = np.stack([f_train[y[train_idx] == c].mean(axis=0) for c in common_cls])

    # Where do the HELD-OUT clips sit?
    held = np.concatenate([same_idx, new_idx])
    f, s = run_big(held, scale)
    f = (f - mean) / std
    dist = np.sort(np.linalg.norm(f[:, None, :] - centres[None], axis=2), axis=1)   # to each centre, nearest first
    pred = np.array(common_cls)[s.argmax(dim=1).numpy()]

    t = meta.loc[held, ["date", "room", "user"]].reset_index(drop=True)
    t["model"] = name
    t["new_room"] = np.arange(len(held)) >= len(same_idx)
    t["d1"], t["d2"], t["norm"] = dist[:, 0], dist[:, 1], np.linalg.norm(f, axis=1)
    t["confidence"] = torch.softmax(s, dim=1).max(dim=1).values.numpy()            # only for the comparison in section 5
    t["wrong"] = (pred != y[held]).astype(np.float32)

    # Write the 3 measurements relative to this model's own same-room clips, so that
    # numbers from different big models mean the same thing.
    same = t[~t.new_room]
    t[INPUTS] = (t[INPUTS] - same[INPUTS].mean()) / same[INPUTS].std()

    train_acc = (np.array(common_cls)[s_train.argmax(dim=1).numpy()] == y[train_idx]).mean()
    new_acc = f"{1 - t.wrong[t.new_room].mean():.3f}" if len(new_idx) else "  -  "
    print(f"  {name:11s}  train acc {train_acc:.3f}   same-room acc {1 - same.wrong.mean():.3f}   new-room acc {new_acc}")
    return t


print("big models (these should match the last epoch printed by the training scripts)")
tables = [measure("every room", "gesture_temporal_resnet_v2.pt", *split_every_room())]
for room in (1, 2, 3):
    tables.append(measure(f"hold out {room}", f"gesture_cross_room{room}_v2.pt", *split_hold_out(room)))
clips = pd.concat(tables, ignore_index=True)


# ----------------------------------------------------------------------------- 4. SMALL MODEL
def train_small(rows):
    """Learn 'is the big model wrong on this clip?' from the 3 measurements."""
    a = torch.tensor(rows[INPUTS].to_numpy(), dtype=torch.float32)
    b = torch.tensor(rows.wrong.to_numpy())
    small = nn.Sequential(nn.Linear(len(INPUTS), 16), nn.ReLU(), nn.Linear(16, 1))
    loss_fn = nn.BCEWithLogitsLoss()                     # yes / no question, so binary cross-entropy
    optimizer = torch.optim.AdamW(small.parameters(), lr=3e-3, weight_decay=0.01)
    for epoch in range(15):
        order = torch.randperm(len(a))
        for i in range(0, len(a), 256):
            j = order[i:i + 256]
            loss = loss_fn(small(a[j]).squeeze(1), b[j])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return small


def risk(small, rows):
    """Chance (0..1) that the big model is wrong, for each clip."""
    with torch.no_grad():
        return torch.sigmoid(small(torch.tensor(rows[INPUTS].to_numpy(), dtype=torch.float32)).squeeze(1)).numpy()


# ----------------------------------------------------------------------------- 5. TEST: PREDICT THE DROP IN A ROOM NEVER USED
summary, sessions = [], []
for room in (1, 2, 3):
    learn = clips[(clips.model != f"hold out {room}") & (clips.room != room)]     # other big models, other rooms
    test = clips[clips.model == f"hold out {room}"].copy()                       # the big model that never saw this room
    small = train_small(learn)
    torch.save(small.state_dict(), f"drop_predictor_room{room}.pt")

    test["risk"] = risk(small, test)
    same, new = test[~test.new_room], test[test.new_room]

    def drops(group):
        """Actual drop (needs labels), and two label-free predictions of it."""
        return pd.Series({"clips": len(group),
                          "actual": group.wrong.mean() - same.wrong.mean(),
                          "small model": group.risk.mean() - same.risk.mean(),
                          "confidence": same.confidence.mean() - group.confidence.mean()})

    whole_room = drops(new)
    per_session = new.groupby(["date", "user"])[["wrong", "risk", "confidence"]].apply(drops)
    summary.append(whole_room.rename(f"room {room}"))
    sessions.append(per_session)

    print(f"\nheld-out room {room}   (small model learned from {len(learn)} clips, {int(learn.wrong.sum())} of them wrong)")
    print(f"  finds the wrong clips: AUROC {roc_auc_score(new.wrong, new.risk):.3f}   (0.5 = guessing, 1.0 = perfect)")
    print(f"  whole room   actual drop {whole_room['actual']:.3f}   predicted {whole_room['small model']:.3f}")
    print("  per recording session (date, user)      clips  actual  predicted")
    for (date, user), r in per_session.iterrows():
        print(f"    {date}  user {user:2d}                 {int(r['clips']):5d}  {r['actual']:6.3f}  {r['small model']:9.3f}")

# ----------------------------------------------------------------------------- 6. RESULTS
summary, sessions = pd.DataFrame(summary), pd.concat(sessions)
print("\naccuracy drop in the unseen room    actual   small model   confidence only")
for name, r in summary.iterrows():
    print(f"  {name}                            {r['actual']:.3f}     {r['small model']:.3f}         {r['confidence']:.3f}")

# How far off is the prediction for one recording session, on average?
# "assume no drop" is the do-nothing answer: expect same-room accuracy everywhere.
# "confidence only" needs no training: the fall in the big model's average confidence.
print(f"\naverage error over the {len(sessions)} recording sessions")
print(f"  assume no drop   {sessions['actual'].abs().mean():.3f}")
print(f"  confidence only  {(sessions['confidence'] - sessions['actual']).abs().mean():.3f}")
print(f"  small model      {(sessions['small model'] - sessions['actual']).abs().mean():.3f}")
print(f"  match between predicted and actual drop (correlation): {np.corrcoef(sessions['small model'], sessions['actual'])[0, 1]:.2f}")
