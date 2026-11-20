"""Model architectures for Widar gesture classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class TemporalResNet(nn.Module):
    """Temporal ResNet v2 for Widar velocity images."""

    def __init__(self, _T, n_classes, width=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(400, width, 3, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.layer1 = TemporalResidualBlock(width, width)
        self.layer2 = TemporalResidualBlock(width, 2 * width, stride=2)
        self.layer3 = TemporalResidualBlock(2 * width, 4 * width, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * width, n_classes),
        )

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class LSTM(nn.Module):
    def __init__(
            self, 
            T,
            n_class, 
            dropout=0.4, 
            lstm_units=128
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5)       # 20x20 -> 16x16 ('valid')
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.drop = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(64, lstm_units, batch_first=True)
        self.out = nn.Linear(lstm_units, n_class)          # logits; softmax lives in the loss
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
        h, _ = self.lstm(x)
        x = self.drop(h[:, -1, :])
        return self.out(x)