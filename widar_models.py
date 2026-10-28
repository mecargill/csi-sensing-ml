"""Model architectures for Widar gesture classification."""
import torch
import torch.nn as nn


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

import torch
import torch.nn as nn


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
