"""CNN architectures and the spectrogram transform, shared by training and the app."""
import numpy as np
import torch.nn as nn
from scipy.signal import spectrogram

from src.config import FS

NPERSEG, NOVERLAP = 128, 96          # 2048-sample window -> 65 freq bins x 61 frames
EPS = 1e-10


# ---------------- 1D-CNN ----------------
def block1d(cin, cout, k, stride=1):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2),
        nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2),
    )


class CNN1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            block1d(1, 16, 64, stride=8),   # wide first kernel, like WDCNN: 2048 -> ~128
            block1d(16, 32, 3),
            block1d(32, 64, 3),
            block1d(64, 64, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------- 2D-CNN ----------------
def to_logspec(W):
    """(N, 2048) windows -> (N, 65, 61) log10 power spectrograms."""
    _, _, S = spectrogram(W, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, axis=-1)
    return np.log10(S + EPS).astype(np.float32)


def block2d(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2),
    )


class CNN2D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            block2d(1, 16),     # 65x61 -> 32x30
            block2d(16, 32),    # -> 16x15
            block2d(32, 64),    # -> 8x7
            block2d(64, 64),    # -> 4x3
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
