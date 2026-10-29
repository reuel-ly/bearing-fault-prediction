"""Training loop, scoring and reporting shared by the three model scripts."""
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEED

EPOCHS, PATIENCE = 60, 10
BATCH, LR, WD = 64, 1e-3, 1e-4
# Seed s=0 is the one checkpointed for the app. It uses the original single-run seed, but the CNN
# runs differ from the original: every DataLoader pass draws from torch's global RNG, so the extra
# per-epoch train-set evaluation shifts the shuffle order. The random forest reproduces exactly.
N_SEEDS = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_seed(k, s):
    return SEED + k + 1000 * s


# ---------------- CNN helpers ----------------
def loader(X, y, shuffle):
    X = torch.from_numpy(X[:, None])                   # add the channel axis
    return DataLoader(TensorDataset(X, torch.from_numpy(y).long()),
                      batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    return torch.cat([model(x.to(DEV)).argmax(1).cpu() for x, _ in dl]).numpy()


@torch.no_grad()
def evaluate(model, dl, loss_fn):
    """Eval-mode mean loss and predictions over a loader."""
    model.eval()
    total, preds = 0.0, []
    for x, y in dl:
        out = model(x.to(DEV))
        total += loss_fn(out, y.to(DEV)).item() * len(y)
        preds.append(out.argmax(1).cpu())
    return total / len(dl.dataset), torch.cat(preds).numpy()


def train_cnn(model, Xtr, ytr, Xva, yva, n_classes, tag):
    """Train with early stopping on validation macro-F1.
    Returns (best state_dict, best F1, per-epoch history)."""
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    dl_tr, dl_va = loader(Xtr, ytr, True), loader(Xva, yva, False)
    dl_tr_eval = loader(Xtr, ytr, False)               # train metrics in eval mode, like validation

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        tr_loss, tr_pred = evaluate(model, dl_tr_eval, loss_fn)
        va_loss, va_pred = evaluate(model, dl_va, loss_fn)
        val_f1 = f1_score(yva, va_pred, average="macro")
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=accuracy_score(ytr, tr_pred),
                         val_loss=va_loss, val_acc=accuracy_score(yva, va_pred), val_f1=val_f1))
        sched.step(val_f1)
        print(f"  {tag} ep {ep:02d} train loss {tr_loss:.4f} | val loss {va_loss:.4f} "
              f"| val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_state, best_f1, hist


def save_history(path, **record):
    path.write_text(json.dumps(record, indent=1))


# ---------------- scoring ----------------
def score(y_true, y_pred, n_classes):
    return dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro"),
        recall=recall_score(y_true, y_pred, average=None, labels=np.arange(n_classes)),
    )


def summarize(results, classes):
    """results[k][s] = score dict. Per round: mean ± std over seeds; overall: mean over all
    runs, ± std over rounds of the seed-averaged scores (how much the test load matters)."""
    acc = np.array([[r["acc"] for r in rk] for rk in results])     # (rounds, seeds)
    f1 = np.array([[r["f1"] for r in rk] for rk in results])
    print()
    for k in range(len(results)):
        print(f"round {k}: acc {acc[k].mean():.4f} ± {acc[k].std(ddof=1):.4f} | "
              f"macro-F1 {f1[k].mean():.4f} ± {f1[k].std(ddof=1):.4f}  (over {acc.shape[1]} seeds)")
    print(f"\nAccuracy : {acc.mean():.4f} ± {acc.mean(1).std(ddof=1):.4f}  (± over rounds)")
    print(f"Macro-F1 : {f1.mean():.4f} ± {f1.mean(1).std(ddof=1):.4f}  (± over rounds)")
    recall = np.mean([r["recall"] for rk in results for r in rk], axis=0)
    for name, rec in zip(classes, recall):
        print(f"  {name:6s} recall {rec:.4f}")
