"""Evaluation figures and tables from the saved test predictions and training histories.
Run from the project root after training:  python -m src.evaluate   -> reports/"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from src.config import ART_DIR, MODELS_DIR, N_LOADS, ROOT
from src.train.common import N_SEEDS

REPORTS = ROOT / "reports"
MODELS = {"1D-CNN": "cnn1d", "2D-CNN": "cnn2d", "Random Forest": "rf"}
CNNS = ["1D-CNN", "2D-CNN"]

# ---------------- style (reference palette, light surface) ----------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL = "#2a78d6", "#eb6834"                  # categorical slots 1-2
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)
CMAP.set_bad(SURFACE)                              # zero cells recede to the surface

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
})


# ---------------- loading ----------------
def classes():
    return list(joblib.load(ART_DIR / "prep_test0.joblib")["le"].classes_)


def preds(sub):
    """All test predictions of one model, pooled over rounds and seeds."""
    ds = [np.load(MODELS_DIR / sub / f"pred_test{k}_seed{s}.npz")
          for k in range(N_LOADS) for s in range(N_SEEDS)]
    return np.concatenate([d["y_true"] for d in ds]), np.concatenate([d["y_pred"] for d in ds])


def histories(sub):
    """h[k][s] = saved history dict."""
    return [[json.loads((MODELS_DIR / sub / f"history_test{k}_seed{s}.json").read_text())
             for s in range(N_SEEDS)] for k in range(N_LOADS)]


def pm(x):
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1):.4f}"


# ---------------- confusion matrices ----------------
def plot_confusion(cms, names, normalized):
    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 6.6))
    fig.subplots_adjust(left=0.05, right=0.92, top=0.86, bottom=0.17, wspace=0.28)
    for ax, (title, cm) in zip(axes, cms.items()):
        M = 100 * cm / cm.sum(1, keepdims=True) if normalized else cm.astype(float)
        vmax = 100 if normalized else M.max()
        img = ax.imshow(np.ma.masked_equal(M, 0), cmap=CMAP,
                        norm=PowerNorm(0.35, vmin=0, vmax=vmax))   # makes small errors visible
        for i, j in zip(*np.nonzero(M)):
            v = M[i, j]
            if normalized:
                txt = f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 0.05 else "<0.1"
            else:
                txt = f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if img.norm(v) > 0.55 else INK)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    unit = "% of true-class windows (rows sum to 100)" if normalized else "window count"
    fig.suptitle(f"Test confusion matrices, pooled over 4 rounds × {N_SEEDS} seeds - {unit}",
                 x=0.01, y=0.98, ha="left", fontsize=12, fontweight="bold")
    fig.colorbar(img, cax=fig.add_axes([0.935, 0.25, 0.01, 0.55])).outline.set_visible(False)
    name = "confusion_normalized.png" if normalized else "confusion_counts.png"
    fig.savefig(REPORTS / name, dpi=150)
    plt.close(fig)


def top_confusions(cm, names, n=3):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0][:n]
    return [(names[i], names[j], int(off[i, j]), 100 * off[i, j] / cm[i].sum())
            for i, j in idx if off[i, j] > 0]


# ---------------- learning curves ----------------
def seed_curve(hk, key, min_seeds=3):
    """Mean and min/max over seeds per epoch, while at least `min_seeds` are still training
    (early stopping ends seeds at different epochs)."""
    runs = [[e[key] for e in h["epochs"]] for h in hk]
    n = max(len(r) for r in runs)
    rows = [[r[i] for r in runs if i < len(r)] for i in range(n)]
    rows = [r for r in rows if len(r) >= min_seeds]
    return (np.arange(len(rows)), np.array([np.mean(r) for r in rows]),
            np.array([np.min(r) for r in rows]), np.array([np.max(r) for r in rows]))


def plot_learning_curves(title, h, path):
    fig, axes = plt.subplots(2, N_LOADS, figsize=(4.2 * N_LOADS, 6.4), sharex="col",
                             constrained_layout=True)
    for k in range(N_LOADS):
        for row, (metric, label) in enumerate([("loss", "Loss (log scale)"), ("acc", "Accuracy")]):
            ax = axes[row, k]
            for split, color, name in [("train", TRAIN, "Train"), ("val", VAL, "Validation")]:
                x, m, lo, hi = seed_curve(h[k], f"{split}_{metric}")
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, m, color=color, label=name)
            if metric == "loss":
                ax.set_yscale("log")
                ax.set_title(f"Round {k} - test load {k}", loc="left")
            if k == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, frameon=False)
    fig.suptitle(f"{title} learning curves - mean over {N_SEEDS} seeds, band = min-max",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------- tables ----------------
def best_epoch(hist):
    return max(hist["epochs"], key=lambda e: e["val_f1"])   # first epoch with the best F1


def write_summary(names, cms, hs):
    L = ["# Evaluation summary", "",
         f"Generated by `python -m src.evaluate`. {N_LOADS} leave-one-load-out rounds × "
         f"{N_SEEDS} seeds per model.", "",
         "## Test accuracy per round (mean ± std over seeds)", "",
         "| Round | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for k in range(N_LOADS):
        L.append(f"| {k} | " + " | ".join(pm([h["test_acc"] for h in hs[m][k]]) for m in MODELS) + " |")
    L += ["", "## Overall (mean over all runs; ± = std over rounds of the seed means)", "",
          "| Model | Test accuracy | Test macro-F1 | Seed std, mean over rounds |", "|---|---|---|---|"]
    for m in MODELS:
        acc = np.array([[h["test_acc"] for h in hk] for hk in hs[m]])
        f1 = np.array([[h["test_f1"] for h in hk] for hk in hs[m]])
        L.append(f"| {m} | {acc.mean():.4f} ± {acc.mean(1).std(ddof=1):.4f} | "
                 f"{f1.mean():.4f} ± {f1.mean(1).std(ddof=1):.4f} | {acc.std(1, ddof=1).mean():.4f} |")

    L += ["", "## CNN generalization gap at the selected (best-validation) epoch", "",
          "Mean ± std over seeds. Train metrics are computed in eval mode, like validation.", "",
          "| Model | Round | Train acc | Val acc | Test acc | Train loss | Val loss |",
          "|---|---|---|---|---|---|---|"]
    for m in CNNS:
        for k in range(N_LOADS):
            b = [best_epoch(h) for h in hs[m][k]]
            L.append(f"| {m} | {k} | {pm([e['train_acc'] for e in b])} | {pm([e['val_acc'] for e in b])} | "
                     f"{pm([h['test_acc'] for h in hs[m][k]])} | {np.mean([e['train_loss'] for e in b]):.4f} | "
                     f"{np.mean([e['val_loss'] for e in b]):.4f} |")

    L += ["", "## Random Forest: train vs out-of-bag vs held-out loads", "",
          "Mean ± std over seeds. OOB windows come from the training loads and overlap their "
          "neighbours by 50 %, so OOB is optimistic.", "",
          "| Round | Train acc | OOB acc | Val acc | Test acc |", "|---|---|---|---|---|"]
    for k in range(N_LOADS):
        hk = hs["Random Forest"][k]
        L.append(f"| {k} | " + " | ".join(pm([h[c] for h in hk])
                                         for c in ("train_acc", "oob_acc", "val_acc", "test_acc")) + " |")

    L += ["", "## Most frequent confusions (pooled test windows)", "",
          "| Model | True → predicted | Windows | % of true class |", "|---|---|---|---|"]
    for m, cm in cms.items():
        for t, p, n, pct in top_confusions(cm, names):
            L.append(f"| {m} | {t} → {p} | {n} | {pct:.1f} % |")

    (REPORTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------- main ----------------
def main():
    REPORTS.mkdir(exist_ok=True)
    names = classes()
    cms = {}
    for m, sub in MODELS.items():
        y, p = preds(sub)
        cms[m] = confusion_matrix(y, p, labels=np.arange(len(names)))
    plot_confusion(cms, names, normalized=True)
    plot_confusion(cms, names, normalized=False)

    hs = {m: histories(sub) for m, sub in MODELS.items()}
    for m in CNNS:
        plot_learning_curves(m, hs[m], REPORTS / f"learning_curves_{MODELS[m]}.png")
    write_summary(names, cms, hs)
    print(f"wrote {', '.join(sorted(p.name for p in REPORTS.iterdir()))} to {REPORTS}")


if __name__ == "__main__":
    main()
