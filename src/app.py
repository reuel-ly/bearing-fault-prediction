"""Bearing fault monitor: dataset statistics + simulated real-time prediction.
Run from the project root:  streamlit run src/app.py
"""
import io
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.io import loadmat
from scipy.signal import decimate
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # streamlit only adds src/
from src.config import ART_DIR, MODELS_DIR, N_LOADS, PAT
from src.features import extract
from src.models import CNN1D, CNN2D, to_logspec
from src.preprocess import filt, load_records, load_round, signature, window
from src.preprocess import main as build_artifacts

# ---------------- config ----------------
MODEL_KINDS = ["1D-CNN", "2D-CNN", "Random Forest"]
CONF_THRESHOLD = 0.6      # below this, show "uncertain" instead of a hard verdict

st.set_page_config(page_title="Bearing fault monitor", layout="wide")


# ---------------- loading helpers ----------------
@st.cache_resource(show_spinner="Building artifacts from data/ (first run only, ~10 s)...")
def ensure_artifacts():
    """artifacts/ is git-ignored, so a fresh clone or deployment builds it on first use."""
    if not all((ART_DIR / f"data_test{k}.npz").exists() for k in range(N_LOADS)):
        build_artifacts()


@st.cache_data
def cached_round(k: int):
    return load_round(k)


@st.cache_resource
def get_records():
    return load_records()


class Predictor:
    """Wraps one trained model and returns class probabilities for (N, 2048) windows."""

    def __init__(self, kind, k):
        self.kind = kind
        meta = joblib.load(ART_DIR / f"prep_test{k}.joblib")
        n = len(meta["le"].classes_)
        if kind == "Random Forest":
            ckpt = joblib.load(MODELS_DIR / "rf" / f"model_test{k}.joblib")
            self.model = ckpt["model"]
        else:
            sub, cls = ("cnn1d", CNN1D) if kind == "1D-CNN" else ("cnn2d", CNN2D)
            ckpt = torch.load(MODELS_DIR / sub / f"model_test{k}.pt", map_location="cpu")
            self.model = cls(n)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            if kind == "2D-CNN":
                self.mu2, self.sd2 = ckpt["mu2"], ckpt["sd2"]
        if ckpt["prep"] != signature(meta):
            raise ValueError(f"model was trained with preprocessing {ckpt['prep']}, but "
                             f"the artifacts use {signature(meta)}")

    def proba(self, W):
        if self.kind == "Random Forest":
            F, _ = extract(W)
            return self.model.predict_proba(F)
        with torch.no_grad():
            if self.kind == "1D-CNN":
                x = torch.from_numpy(W[:, None, :].astype(np.float32))
            else:
                S = (to_logspec(W) - self.mu2) / self.sd2
                x = torch.from_numpy(S[:, None, :, :].astype(np.float32))
            return torch.softmax(self.model(x), dim=1).numpy()


@st.cache_resource
def get_predictor(kind, k):
    return Predictor(kind, k)


def preprocess_signal(x, meta, is_48k):
    """Raw DE signal -> normalized windows, using the same settings as the training round."""
    if is_48k:
        x = decimate(x, 4)
    x = filt(x, meta["lowpass"], meta["bandpass"], meta["band"], meta["fs"])
    W = window(x, meta["size"], meta["size"])
    W = W - W.mean(1, keepdims=True)
    return ((W - meta["mu"]) / meta["sd"]).astype(np.float32)


def wave_df(w, fs):
    return pd.DataFrame({"amplitude": w}, index=np.arange(len(w)) / fs)


# ---------------- tab 1: dataset statistics ----------------
def dataset_tab(data, meta, classes):
    fs, size = meta["fs"], meta["size"]
    train_loads = sorted(set(range(N_LOADS)) - {meta["test_load"], meta["val_load"]})
    st.subheader("Windows per class in this round")
    st.caption(f"Train loads {train_loads} (50% overlap) | validation load {meta['val_load']} | "
               f"test load {meta['test_load']} (no overlap)")

    names = {"ytr": "Train", "yva": "Validation", "yte": "Test"}
    counts = pd.DataFrame({
        label: pd.Series(classes[data[key]]).value_counts().reindex(classes).fillna(0).astype(int)
        for key, label in names.items()
    })
    c1, c2 = st.columns([2, 1])
    c1.bar_chart(counts)
    c2.dataframe(counts, width="stretch")

    st.subheader("Per-class signal statistics (training windows)")
    st.caption("Values are in normalized units (after the shared mean/std scaling).")
    Wtr, ytr = data["Wtr"], data["ytr"]
    stats = pd.DataFrame({
        "label": classes[ytr],
        "rms": np.sqrt((Wtr ** 2).mean(1)),
        "kurtosis": kurtosis(Wtr, axis=1),
    }).groupby("label").mean().reindex(classes)
    c1, c2 = st.columns(2)
    c1.markdown("**Mean RMS**")
    c1.bar_chart(stats["rms"])
    c2.markdown("**Mean kurtosis** (higher = more impulsive)")
    c2.bar_chart(stats["kurtosis"])

    st.subheader("Example window")
    cls = st.selectbox("Class", classes, key="ex_class")
    pool = np.where(ytr == list(classes).index(cls))[0]
    i = st.slider("Window #", 0, len(pool) - 1, 0, key="ex_idx")
    w = Wtr[pool[i]]
    c1, c2 = st.columns(2)
    c1.markdown("**Waveform**")
    c1.line_chart(wave_df(w, fs), height=250)
    freqs = np.fft.rfftfreq(size, 1 / fs)
    spec = np.abs(np.fft.rfft(w * np.hanning(size)))
    c2.markdown("**Spectrum**")
    c2.line_chart(pd.DataFrame({"magnitude": spec}, index=freqs), height=250)

    st.subheader("Mean spectrum per class")
    default = [c for c in ("Normal", "IR007", "B007") if c in classes]
    chosen = st.multiselect("Classes to compare", list(classes), default=default)
    if chosen:
        hann = np.hanning(size)
        cols = {}
        for c in chosen:
            sel = Wtr[ytr == list(classes).index(c)][:300]
            cols[c] = np.abs(np.fft.rfft(sel * hann, axis=1)).mean(0)
        st.line_chart(pd.DataFrame(cols, index=freqs), height=300)

    st.subheader("Raw records")
    if st.checkbox("Load raw records (reads all .mat files, takes a moment)"):
        try:
            recs = get_records()
        except Exception as e:                       # noqa: BLE001
            st.error(f"Could not load records via src/preprocess.py: {e}")
            return
        df = pd.DataFrame([
            dict(label=r["label"], fault=r["fault"], diam=r["diam"], pos=r["pos"],
                 load=r["load"], samples=len(r["x"]), seconds=round(len(r["x"]) / fs, 1),
                 rms=float(np.sqrt((r["x"] ** 2).mean())), kurtosis=float(kurtosis(r["x"])))
            for r in recs
        ])
        st.markdown("**Records per class and load**")
        st.dataframe(df.pivot_table(index="label", columns="load", values="samples",
                                    aggfunc="count", fill_value=0), width="stretch")
        st.markdown("**Mean RMS and kurtosis per class (raw acceleration units)**")
        st.dataframe(df.groupby("label")[["rms", "kurtosis"]].mean(), width="stretch")
        with st.expander("All records"):
            st.dataframe(df, width="stretch")


# ---------------- tab 2: live prediction ----------------
def live_tab(data, meta, classes, kind, k):
    fs = meta["fs"]
    src = st.radio("Signal source", ["Held-out test windows", "Upload a .mat file"], horizontal=True)

    if src == "Held-out test windows":
        W, y = data["Wte"], data["yte"]
        choice = st.selectbox("Simulate machine condition", ["All (shuffled)"] + list(classes))
        if choice == "All (shuffled)":
            idx = np.random.default_rng(0).permutation(len(y))
        else:
            idx = np.where(y == list(classes).index(choice))[0]
        Wsrc, truth = W[idx], classes[y[idx]]
    else:
        up = st.file_uploader("CWRU drive-end .mat file", type="mat")
        if up is None:
            st.info("Upload a file such as `IR007_3.mat` or `Normal_0.mat`.")
            return
        d = loadmat(io.BytesIO(up.getvalue()))
        key = next((kk for kk in d if kk.endswith("DE_time")), None)
        if key is None:
            st.error("No `*DE_time` channel found in this file.")
            return
        x = d[key].squeeze()
        is_48k = st.checkbox("File is 48 kHz (decimate to 12 kHz)",
                             value=up.name.startswith("Normal") or len(x) > 300_000)
        Wsrc = preprocess_signal(x, meta, is_48k)
        truth = None
        m = PAT.match(up.name)
        if m:
            fault, diam, _, load = m.groups()
            label = fault + (diam or "")
            if label in classes:
                truth = np.array([label] * len(Wsrc))
            status = ("held-out for this round" if int(load) == meta["test_load"]
                      else "was in this round's train/validation data, so results are optimistic")
            st.caption(f"Load {load} {status}.")
        else:
            st.caption("Filename doesn't match the CWRU pattern, so there is no true label.")

    c1, c2 = st.columns(2)
    n_stream = int(c1.number_input("Windows to stream", 1, len(Wsrc), min(len(Wsrc), 200)))
    delay = c2.slider("Delay per window (s)", 0.0, 1.0, 0.17, 0.01,
                      help=f"One window is {meta['size'] / fs:.2f} s of sensor data.")

    try:
        predictor = get_predictor(kind, k)
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}. Train the {kind} for round {k} first.")
        return
    except (KeyError, TypeError, ValueError) as e:
        st.error(f"The saved {kind} for round {k} doesn't match the current artifacts ({e}). "
                 "Retrain it after running `python -m src.preprocess`.")
        return

    # --- playback state survives reruns ---
    st.session_state.setdefault("cur", 0)
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("jump_to", 0)
    st.session_state.cur = min(st.session_state.cur, n_stream - 1)
    st.session_state.jump_to = min(st.session_state.jump_to, n_stream - 1)
    normal_idx = list(classes).index("Normal")

    def play():
        if st.session_state.cur >= n_stream - 1:      # finished before: start over
            st.session_state.cur = 0
        st.session_state.playing = True

    def pause():
        st.session_state.playing = False

    def step(delta):
        st.session_state.cur = int(np.clip(st.session_state.cur + delta, 0, n_stream - 1))
        st.session_state.playing = False

    def jump():
        st.session_state.cur = min(int(st.session_state.jump_to), n_stream - 1)
        st.session_state.playing = False

    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
    b1.button("▶ Play", on_click=play, width="stretch")
    b2.button("⏸ Pause", on_click=pause, width="stretch")
    b3.button("◀ Prev", on_click=step, args=(-1,), width="stretch")
    b4.button("Next ▶", on_click=step, args=(1,), width="stretch")
    b5.number_input("Go to window", 0, n_stream - 1, key="jump_to", on_change=jump)

    alert_ph, progress_ph = st.empty(), st.empty()
    left, right = st.columns([2, 1])
    wave_ph, prob_ph = left.empty(), right.empty()
    acc_ph, table_ph = st.empty(), st.empty()

    def render(i, p, w):
        pred = classes[int(p.argmax())]
        p_normal = p[normal_idx]
        p_fault = 1 - p_normal
        top = np.argsort(p)[::-1][:3]
        top_txt = ", ".join(f"{classes[j]} {p[j]:.0%}" for j in top)

        # step 1: normal vs fault (probabilities of all fault classes are added up)
        if p_normal >= CONF_THRESHOLD:
            alert_ph.success(f"Window {i}: Normal ({p_normal:.0%})")
        elif p_fault >= CONF_THRESHOLD:
            # step 2: which fault type? only name one if a single class is confident
            if p.max() >= CONF_THRESHOLD:
                alert_ph.error(f"Window {i}: fault detected, {pred} ({p.max():.0%})")
            else:
                alert_ph.warning(f"Window {i}: fault detected ({p_fault:.0%}), "
                                 f"type uncertain. Top guesses: {top_txt}")
        else:
            alert_ph.warning(f"Window {i}: uncertain (fault chance {p_fault:.0%}). "
                             f"Top: {top_txt}")
        wave_ph.line_chart(wave_df(w[0], fs), height=250)
        prob_ph.bar_chart(pd.Series(p, index=classes), height=250)
        progress_ph.progress((i + 1) / n_stream, text=f"Window {i} of {n_stream - 1}")
        return pred

    # --- paused: show the window at the saved position ---
    if not st.session_state.playing:
        i = st.session_state.cur
        w = Wsrc[i:i + 1]
        render(i, predictor.proba(w)[0], w)
        msg = f"Paused at window {i}."
        if truth is not None:
            msg += f" True label: {truth[i]}."
        st.caption(msg)
        return

    # --- playing: resume from the saved position ---
    hist, correct = [], 0
    for i in range(st.session_state.cur, n_stream):
        st.session_state.cur = i
        w = Wsrc[i:i + 1]
        p = predictor.proba(w)[0]
        pred = render(i, p, w)

        row = dict(window=i, predicted=pred, confidence=round(float(p.max()), 3))
        if truth is not None:
            row["true"] = truth[i]
            correct += int(pred == truth[i])
        hist.append(row)

        if truth is not None:
            acc_ph.metric(f"Running accuracy ({len(hist)} windows)", f"{correct / len(hist):.1%}")
        table_ph.dataframe(pd.DataFrame(hist).tail(12).iloc[::-1], hide_index=True,
                           width="stretch")
        time.sleep(delay)

    st.session_state.playing = False                  # reached the end: stop on the last window
    st.session_state.cur = n_stream - 1


# ---------------- main ----------------
st.title("Bearing fault monitor")

with st.sidebar:
    st.header("Settings")
    k = st.selectbox("Round (held-out test load)", list(range(N_LOADS)))
    kind = st.selectbox("Model", MODEL_KINDS)

ensure_artifacts()
data, meta = cached_round(k)
classes = np.array(meta["le"].classes_)

tab_stats, tab_live = st.tabs(["Dataset statistics", "Live prediction"])
with tab_stats:
    dataset_tab(data, meta, classes)
with tab_live:
    live_tab(data, meta, classes, kind, k)
