"""Shared CWRU preprocessing: load, filter, window, split by load, normalize, save.
Run from the project root:  python -m src.preprocess"""
from collections import Counter

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, decimate, ellip, sosfiltfilt
from sklearn.preprocessing import LabelEncoder

from src.config import (ART_DIR, BAND, DE_ROOT, FS, LOWPASS, N_LOADS, NORMAL_ROOT, PAT, SIZE,
                        USE_BANDPASS)

LP_SOS = ellip(8, 0.1, 80, LOWPASS, fs=FS, output="sos")


# ---------------- loading ----------------
def load_de(fp):
    d = loadmat(fp)
    return d[next(k for k in d if k.endswith("DE_time"))].squeeze()


def load_records():
    recs = []
    for base in (DE_ROOT, NORMAL_ROOT):
        for f in sorted(base.rglob("*.mat")):
            m = PAT.match(f.name)
            if not m:
                print("skipped:", f.name)
                continue
            fault, diam, pos, load = m.groups()
            if diam == "028":                 # no OR028, would break class consistency
                continue
            x = load_de(f)
            if fault == "Normal":
                x = decimate(x, 4)            # 48 kHz -> 12 kHz
            recs.append(dict(fault=fault, diam=diam, pos=pos, load=int(load),
                             x=x, label=fault + (diam or "")))

    assert recs, "no files found, check DE_ROOT / NORMAL_ROOT"
    assert all(len(r["x"]) < 300_000 for r in recs), "a 48 kHz record slipped through"
    return recs


# ---------------- DSP ----------------
def filt(x, lowpass=LOWPASS, bandpass=USE_BANDPASS, band=BAND, fs=FS):
    """Common low-pass for every record, plus the optional band-pass."""
    sos = LP_SOS if lowpass == LOWPASS else ellip(8, 0.1, 80, lowpass, fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    if bandpass:
        x = sosfiltfilt(butter(4, list(band), btype="band", fs=fs, output="sos"), x)
    return x


def window(x, size, step):
    n = (len(x) - size) // step + 1
    return np.stack([x[i * step : i * step + size] for i in range(n)])


def prep(recs, step):
    """Filter, window, and remove each window's mean."""
    Ws, ys = [], []
    for r in recs:
        w = window(filt(r["x"]), SIZE, step)
        w = w - w.mean(1, keepdims=True)
        Ws.append(w)
        ys += [r["label"]] * len(w)
    return np.concatenate(Ws), np.array(ys)


def signature(meta):
    """Preprocessing settings a model depends on; saved with every checkpoint."""
    return {k: meta[k] for k in ("fs", "size", "lowpass", "bandpass", "band")}


def load_round(k):
    d = np.load(ART_DIR / f"data_test{k}.npz")
    data = {n: d[n] for n in ("Wtr", "ytr", "Wva", "yva", "Wte", "yte")}
    return data, joblib.load(ART_DIR / f"prep_test{k}.joblib")


# ---------------- one round ----------------
def build_round(recs, test_load, le, classes):
    val_load = (test_load + 1) % N_LOADS
    tr = [r for r in recs if r["load"] not in (test_load, val_load)]
    va = [r for r in recs if r["load"] == val_load]
    te = [r for r in recs if r["load"] == test_load]

    Wtr, ytr = prep(tr, SIZE // 2)            # 50% overlap, train only
    Wva, yva = prep(va, SIZE)                 # no overlap
    Wte, yte = prep(te, SIZE)                 # no overlap

    assert set(ytr) == set(yva) == set(yte) == set(classes), f"class mismatch in round {test_load}"

    mu, sd = Wtr.mean(), Wtr.std()            # train statistics only
    Wtr, Wva, Wte = [((W - mu) / sd).astype(np.float32) for W in (Wtr, Wva, Wte)]

    np.savez_compressed(
        ART_DIR / f"data_test{test_load}.npz",
        Wtr=Wtr, ytr=le.transform(ytr),
        Wva=Wva, yva=le.transform(yva),
        Wte=Wte, yte=le.transform(yte),
    )
    joblib.dump(
        dict(mu=mu, sd=sd, le=le, test_load=test_load, val_load=val_load,
             fs=FS, size=SIZE, lowpass=LOWPASS, bandpass=USE_BANDPASS, band=BAND),
        ART_DIR / f"prep_test{test_load}.joblib",
    )
    print(test_load, Wtr.shape, Wva.shape, Wte.shape)


# ---------------- main ----------------
def main():
    ART_DIR.mkdir(exist_ok=True)
    recs = load_records()
    print(Counter(r["label"] for r in recs))

    classes = sorted({r["label"] for r in recs})
    le = LabelEncoder().fit(classes)          # same encoder for every round

    for test_load in range(N_LOADS):
        build_round(recs, test_load, le, classes)


if __name__ == "__main__":
    main()
