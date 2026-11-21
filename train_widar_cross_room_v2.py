"""
Version 2 of train_widar_cross_room.py: the same room hold-out test, using the data
changes from train_widar_temporal_resnet_v2.py (drop the suspect "Slide" session,
square-root scaling, train on all 22 gestures).

One whole room is held out. The model trains on the other two rooms and is tested on
the room it never saw. 10% of the training rooms is also held out, so

  same-room accuracy - new-room accuracy = the domain gap.

The extra-gesture training clips come from the TRAINING rooms only. Nothing recorded
in the held-out room is ever trained on.

  python3 train_widar_cross_room_v2.py 3      (hold out room 1, 2 or 3; default 3)
"""
import sys

import numpy as np
import torch
import torch.nn as nn

import widar_data

torch.manual_seed(0)

TEST_ROOM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
assert TEST_ROOM in (1, 2, 3), "Widar has rooms 1, 2 and 3"

# ----------------------------------------------------------------------------- 1. DATA
COMMON = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(H)", "Draw-Zigzag(H)"]

meta = widar_data.load_metadata()
suspect = (meta.date == "20181109") & (meta.gesture == "Slide")
meta = meta[~suspect].reset_index(drop=True)             # drop the suspect "Slide" session

X, y, classes = widar_data.load_arrays(meta)             # every gesture, not only COMMON


common_cls = [classes.index(g) for g in COMMON]          # the 6 output columns we are graded on
is_common = np.isin(y, common_cls)
in_test_room = (meta.room == TEST_ROOM).to_numpy()

# Split by room instead of at random.
rng = np.random.default_rng(0)
other_rooms = rng.permutation(np.flatnonzero(is_common & ~in_test_room))
n_train = int(0.9 * len(other_rooms))
train_idx = other_rooms[:n_train]                        # 90% of the two training rooms
same_room_idx = other_rooms[n_train:]                    # 10% of the training rooms, never trained on
new_room_idx = np.flatnonzero(is_common & in_test_room)  # every 6-gesture clip from the held-out room
extra_idx = np.flatnonzero(~is_common & ~in_test_room)   # other 16 gestures, training rooms only
fit_idx = np.concatenate([train_idx, extra_idx])         # everything the model trains on
print(f"hold out room {TEST_ROOM}:  train {len(train_idx)} + {len(extra_idx)} extra-gesture clips   "
      f"same-room test {len(same_room_idx)}   new-room test {len(new_room_idx)}")

# Scale to [0, 1] using training clips only (nothing leaks from the new room), then square root.
scale = X.reshape(len(X), -1).max(axis=1)[fit_idx].max()
X = np.sqrt(np.clip(X, 0, None) / scale)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
X_t = torch.tensor(X, device=device)
y_t = torch.tensor(y, device=device)


# ----------------------------------------------------------------------------- 1b. AUGMENTATION (unchanged)
def random_shift(x, dim, lo, hi):
    """Slide every sample along one axis by its own random amount (lo..hi steps); emptied slots become 0."""
    B, n = len(x), x.shape[dim]
    k = torch.randint(lo, hi + 1, (B, 1), device=x.device)
    src = torch.arange(n, device=x.device) - k                # (B, n): which old position lands in each slot
    shape = [B, 1, 1, 1]
    shape[dim] = n                                            # so src lines up with the axis being shifted
    inside = ((src >= 0) & (src < n)).view(shape)             # slots whose source fell off the edge stay empty
    return x.gather(dim, src.clamp(0, n - 1).view(shape).expand_as(x)) * inside


def augment(x):
    """Return a randomly shifted copy of a training batch x: (B, 30, 20, 20)."""
    x = random_shift(x, 1, 0, 6)     # time: start 0..6 frames later
    x = random_shift(x, 2, -1, 1)    # velocity image: one pixel either way, on each axis
    x = random_shift(x, 3, -1, 1)
    return x


# ----------------------------------------------------------------------------- 2. MODEL (unchanged, but 22 outputs)
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

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)             # (N, 30, 20, 20) -> (N, 400, 30)
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


model = TemporalResNet(n_classes=len(classes)).to(device)     # one output per gesture: 22

# ----------------------------------------------------------------------------- 3 + 4. LOSS, OPTIMIZER (unchanged)
BATCH, EPOCHS = 64, 30
steps_per_epoch = (len(fit_idx) + BATCH - 1) // BATCH

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=EPOCHS * steps_per_epoch)


def predict(indices):
    """Predicted class for each sample. Only the 6 common gestures compete."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), 512):
            scores = model(X_t[indices[i:i + 512]])[:, common_cls]    # keep the 6 columns we are graded on
            out.append(scores.argmax(dim=1))
    model.train()
    return np.array(common_cls)[torch.cat(out).cpu().numpy()]         # column 0..5 -> class number


def accuracy(indices):
    return (predict(indices) == y[indices]).mean()


# ----------------------------------------------------------------------------- 5. TRAINING LOOP
for epoch in range(EPOCHS):
    rng.shuffle(fit_idx)
    total_loss = 0
    for i in range(0, len(fit_idx), BATCH):
        b = fit_idx[i:i + BATCH]
        loss = loss_fn(model(augment(X_t[b])), y_t[b])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(b)
    print(f"epoch {epoch + 1:2d}  loss {total_loss / len(fit_idx):.3f}  "
          f"same-room acc {accuracy(same_room_idx):.3f}  new-room acc {accuracy(new_room_idx):.3f}")

torch.save(model.state_dict(), f"results/gesture_cross_room{TEST_ROOM}_v2.pt")

# ----------------------------------------------------------------------------- 6. RESULTS
same_pred, same_true = predict(same_room_idx), y[same_room_idx]
new_pred, new_true = predict(new_room_idx), y[new_room_idx]
same_acc, new_acc = (same_pred == same_true).mean(), (new_pred == new_true).mean()

print(f"\nheld-out room {TEST_ROOM}")
print(f"  same-room accuracy {same_acc:.3f}   (rooms the model trained in)")
print(f"  new-room accuracy  {new_acc:.3f}   (room it never saw)")
print(f"  domain gap         {same_acc - new_acc:.3f}")

print("\nper gesture         same-room  new-room")
for c in sorted(common_cls):
    if (new_true == c).any():                        # a room may be missing a gesture once clips are dropped
        print(f"  {classes[c]:16s}  {(same_pred[same_true == c] == c).mean():.3f}      {(new_pred[new_true == c] == c).mean():.3f}")

# A new room can also mean new PEOPLE. Splitting the new-room score by person shows
# how much of the drop is the room and how much is someone the model has never seen.
train_users = set(meta.user[fit_idx])
new_users = meta.user[new_room_idx].to_numpy()
print("\nnew-room accuracy per person")
for u in sorted(set(new_users)):
    m = new_users == u
    seen = "also in the training rooms" if u in train_users else "never seen in training"
    print(f"  user {u:2d}  {(new_pred[m] == new_true[m]).mean():.3f}   n={m.sum():4d}   {seen}")
