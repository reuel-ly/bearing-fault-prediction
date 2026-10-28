"""Hand-crafted vibration features for the Random Forest, shared by training and the app."""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from src.config import FS, LOWPASS, SIZE

FREQS = np.fft.rfftfreq(SIZE, 1 / FS)
HANN = np.hanning(SIZE)
SPEC_EDGES = np.arange(0, LOWPASS + 1, 500)   # 500 Hz bands up to the common low-pass cutoff
ENV_EDGES = np.arange(0, 601, 50)             # 12 bands of 50 Hz (covers the bearing fault frequencies)
EPS = 1e-12


def band_energy_ratio(P, edges):
    """Fraction of total energy in each band. P is a power spectrum, shape (N, bins)."""
    cols = [P[:, (FREQS >= lo) & (FREQS < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])]
    B = np.stack(cols, axis=1)
    return B / (B.sum(1, keepdims=True) + EPS)


def extract(W):
    """W: (N, 2048) windows -> (N, n_features) feature matrix, plus feature names."""
    A = np.abs(W)
    rms = np.sqrt((W ** 2).mean(1))
    peak = A.max(1)
    mean_abs = A.mean(1)
    sra = np.sqrt(A).mean(1) ** 2
    time_f = {
        "rms": rms,
        "peak": peak,
        "p2p": W.max(1) - W.min(1),
        "kurtosis": kurtosis(W, axis=1),
        "skewness": skew(W, axis=1),
        "crest": peak / (rms + EPS),
        "shape": rms / (mean_abs + EPS),
        "impulse": peak / (mean_abs + EPS),
        "margin": peak / (sra + EPS),
    }

    # spectrum of the signal
    P = np.abs(np.fft.rfft(W * HANN, axis=1)) ** 2
    centroid = (P * FREQS).sum(1) / (P.sum(1) + EPS)
    spec_bands = band_energy_ratio(P, SPEC_EDGES)

    # envelope spectrum (amplitude modulation at the fault frequencies)
    env = np.abs(hilbert(W, axis=1))
    env = env - env.mean(1, keepdims=True)
    PE = np.abs(np.fft.rfft(env * HANN, axis=1)) ** 2
    env_bands = band_energy_ratio(PE, ENV_EDGES)

    names = list(time_f) + ["spec_centroid"]
    names += [f"spec_{lo}-{hi}Hz" for lo, hi in zip(SPEC_EDGES[:-1], SPEC_EDGES[1:])]
    names += [f"env_{lo}-{hi}Hz" for lo, hi in zip(ENV_EDGES[:-1], ENV_EDGES[1:])]

    F = np.column_stack([*time_f.values(), centroid, spec_bands, env_bands])
    return F.astype(np.float32), names
