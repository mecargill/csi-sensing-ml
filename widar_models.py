"""Model architectures for Widar gesture classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureCNN(nn.Module):
    """Two convolutional layers followed by a class-scoring head."""

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 5 * 5, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Convolutional block with a skip connection."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class GestureResNet(nn.Module):
    """Small residual network for Widar velocity images."""

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class CNNGRU(nn.Module):
    """Per-frame CNN encoder followed by a GRU over time.

    Input:  [B, T, H, W]  (time on axis 1, one BVP frame per step)
    Output: [B, num_classes] logits
    """

    def __init__(
        self,
        T: int,                      # unused; kept to match model_factory(X.shape[1], n_classes)
        num_classes: int,
        feat_dim: int = 128,
        hidden: int = 64,
        gru_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Frame encoder: shared across all time steps
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),          # -> [N, 64, 2, 2], works for any frame size
            nn.Flatten(),                      # -> [N, 256]
            nn.Linear(256, feat_dim), nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal model
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional

        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape

        # Fold time into the batch so the CNN sees every frame independently
        f = self.cnn(x.reshape(B * T, 1, H, W))      # [B*T, feat_dim]
        f = f.reshape(B, T, -1)                      # [B, T, feat_dim]

        _, h_n = self.gru(f)                         # h_n: [layers * dirs, B, hidden]
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # last layer, forward + backward
        else:
            h = h_n[-1]
        return self.head(h)

class KerasGRU(nn.Module):
    """GRU matching Keras (<=2.3) defaults: reset_after=False, sigmoid gates.
    n = tanh(W x + U (r * h) + b)      <- reset applied BEFORE the recurrent matmul
    h = z * h_prev + (1 - z) * n
    nn.GRU applies the reset gate after the matmul, so it is NOT equivalent.
    """
    def __init__(
            self, input_size, hidden_size
            ):
        super().__init__()
        self.hs = hidden_size
        self.W = nn.Parameter(torch.empty(input_size, 3 * hidden_size))   # kernel (z, r, h)
        self.U = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))  # recurrent kernel
        self.b = nn.Parameter(torch.zeros(3 * hidden_size))
        nn.init.xavier_uniform_(self.W)                                    # glorot_uniform
        for i in range(3):                                                 # orthogonal per gate
            nn.init.orthogonal_(self.U.data[:, i * hidden_size:(i + 1) * hidden_size])
 
    def forward(self, x):  # x: (B, T, F) -> last hidden state (B, H)
        B, T, _ = x.shape
        H = self.hs
        h = x.new_zeros(B, H)
        xw = x @ self.W + self.b                      # (B, T, 3H)
        Uz, Ur, Uh = self.U[:, :H], self.U[:, H:2 * H], self.U[:, 2 * H:]
        for t in range(T):
            xz, xr, xh = xw[:, t].split(H, dim=-1)
            z = torch.sigmoid(xz + h @ Uz)
            r = torch.sigmoid(xr + h @ Ur)
            n = torch.tanh(xh + (r * h) @ Uh)
            h = z * h + (1 - z) * n
        return h
 
 
class Widar3Net(nn.Module):
    def __init__(
            self, 
            T,
            n_class, 
            dropout=0.5, 
            gru_units=128
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5)       # 20x20 -> 16x16 ('valid')
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.drop = nn.Dropout(dropout)
        #self.gru = KerasGRU(64, gru_units)
        self.gru = nn.GRU(64, gru_units, batch_first=True)
        self.out = nn.Linear(gru_units, n_class)          # logits; softmax lives in the loss
        for m in (self.conv, self.fc1, self.fc2, self.out):   # Keras init: glorot_uniform, zero bias
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
 
    def forward(self, x):
        # x: (B, T, 1, 20, 20)  [channels-first; Keras was (B, T, 20, 20, 1)]
        B, T = x.shape[:2]
        x = x.reshape(B * T, 1, 20, 20)                   # TimeDistributed == fold time into batch
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = x.reshape(B, T, -1)
        _, h = self.gru(x)
        x = self.drop(h[-1])
        return self.out(x)
 
 
    def make_optimizer(model, lr=3e-4):
        # Keras RMSprop: rho=0.9, epsilon=1e-7 (added outside the sqrt, same as PyTorch)
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-7)