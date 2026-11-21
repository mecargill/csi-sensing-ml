"""
Third model: a ResNet that convolves over TIME, for Widar 3.0 gestures.

train_widar_resnet.py treated the 30 time frames as 30 image channels. Its first
conv layer then has one fixed set of weights per frame number, so "the hand moved
at frame 3" and "the hand moved at frame 7" look like unrelated inputs. It has to
memorize every possible timing, which is why it overfit (96% train, 82% test).

A gesture is a motion over time, so here the convolution slides along the time
axis instead. Three changes from train_widar_resnet.py:

  1. MODEL    Conv2d over the 20x20 image  ->  Conv1d over the 30 frames.
              Each frame's 400 pixels are the input channels. The residual block
              is the same idea as before.
  2. DATA     Random shifts of every training batch (augmentation). We already
              use every sample Widar has for these gestures, so this is how we
              get "more data" for free.
  3. TRAINING Weight decay, a learning-rate schedule, and 30 epochs instead of 10.
"""
import numpy as np
import torch
import torch.nn as nn

import widar_data

torch.manual_seed(0)

# ----------------------------------------------------------------------------- 1. DATA (unchanged)
COMMON = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(H)", "Draw-Zigzag(H)"]

meta = widar_data.load_metadata()
meta = meta[meta.gesture.isin(COMMON)].reset_index(drop=True)
print(f"{len(meta)} samples of {len(COMMON)} gestures")

X, y, classes = widar_data.load_arrays(meta)
X = X / (X.max() + 1e-8)      # (N, 30, 20, 20) = (sample, time, vx, vy), scaled to [0, 1]

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
n_train = int(0.8 * len(X))
train_idx, test_idx = idx[:n_train], idx[n_train:]   # same split as the other two scripts

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
X_t = torch.tensor(X, device=device)
y_t = torch.tensor(y, device=device)


# ----------------------------------------------------------------------------- 1b. AUGMENTATION (new)
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


def augment(x):
    """Return a randomly shifted copy of a training batch x: (B, 30, 20, 20).

    The label does not change if the gesture starts a little later or the velocity
    image moves by one pixel, so the model sees a slightly different version of
    every sample each epoch and cannot simply memorize them.
    """
    x = random_shift(x, 1, 0, 6)     # time: start 0..6 frames later (clips are ~16 frames, zero-padded to 30)
    x = random_shift(x, 2, -1, 1)    # velocity image: one pixel either way, on each axis
    x = random_shift(x, 3, -1, 1)
    return x


# ----------------------------------------------------------------------------- 2. MODEL
class ResidualBlock(nn.Module):
    """Same block as before, with Conv1d (slides over time) instead of Conv2d (slides over the image)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),   # looks at 3 neighbouring frames
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
        return torch.relu(self.body(x) + self.skip(x))   # <-- the residual connection


class TemporalResNet(nn.Module):
    def __init__(self, n_classes, width=128):
        super().__init__()
        self.stem = nn.Sequential(                                        # (N,400,30) -> (N,128,30)
            nn.Conv1d(400, width, 3, padding=1, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        self.layer1 = ResidualBlock(width, width)                         # -> (N,128,30)
        self.layer2 = ResidualBlock(width, 2 * width, stride=2)           # -> (N,256,15)
        self.layer3 = ResidualBlock(2 * width, 4 * width, stride=2)       # -> (N,512,8)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                                      # average over time -> (N,512,1)
            nn.Flatten(),                                                 # -> (N,512)
            nn.Dropout(0.3),
            nn.Linear(4 * width, n_classes),                              # -> (N,n_classes)
        )

    def forward(self, x):
        # (N, 30, 20, 20) -> (N, 30, 400) -> (N, 400, 30): Conv1d wants (batch, channels, time).
        x = x.flatten(2).transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


model = TemporalResNet(n_classes=len(classes)).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

# ----------------------------------------------------------------------------- 3 + 4. LOSS, OPTIMIZER
BATCH, EPOCHS = 64, 100
steps_per_epoch = (len(train_idx) + BATCH - 1) // BATCH

loss_fn = nn.CrossEntropyLoss()
# AdamW = Adam + weight decay. Weight decay pulls every weight toward zero a little each step,
# which discourages the huge, very specific weights that memorization needs.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
# The schedule ramps the learning rate up, then slowly down to ~0, so the last epochs
# make small careful steps and the final model is stable instead of bouncing around.
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=EPOCHS * steps_per_epoch)


def predict(indices):
    """Predicted class for each sample. No augmentation here: we test on the real data."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), 512):
            out.append(model(X_t[indices[i:i + 512]]).argmax(dim=1))
    model.train()
    return torch.cat(out).cpu().numpy()


def accuracy(indices):
    return (predict(indices) == y[indices]).mean()


# ----------------------------------------------------------------------------- 5. TRAINING LOOP
for epoch in range(EPOCHS):
    rng.shuffle(train_idx)
    total_loss = 0
    for i in range(0, len(train_idx), BATCH):
        b = train_idx[i:i + BATCH]
        loss = loss_fn(model(augment(X_t[b])), y_t[b])   # <-- train on the shifted copy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()                                 # <-- update the learning rate every step
        total_loss += loss.item() * len(b)
    print(f"epoch {epoch + 1:2d}  loss {total_loss / len(train_idx):.3f}  "
          f"train acc {accuracy(train_idx):.3f}  test acc {accuracy(test_idx):.3f}")

torch.save(model.state_dict(), "results/gesture_temporal_resnet.pt")

# ----------------------------------------------------------------------------- 6. RESULTS
pred, true = predict(test_idx), y[test_idx]
print(f"\nfinal test accuracy {(pred == true).mean():.3f}")
for c, name in enumerate(classes):
    print(f"  {name:16s} {(pred[true == c] == c).mean():.3f}")
