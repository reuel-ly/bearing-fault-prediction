"""Training loop, scoring and reporting shared by the three model scripts."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

EPOCHS, PATIENCE = 60, 10
BATCH, LR, WD = 64, 1e-3, 1e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------- CNN helpers ----------------
def loader(X, y, shuffle):
    X = torch.from_numpy(X[:, None])                   # add the channel axis
    return DataLoader(TensorDataset(X, torch.from_numpy(y).long()),
                      batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    return torch.cat([model(x.to(DEV)).argmax(1).cpu() for x, _ in dl]).numpy()


def train_cnn(model, Xtr, ytr, Xva, yva, n_classes, tag):
    """Train with early stopping on validation macro-F1; returns (best state_dict, best F1)."""
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    dl_tr, dl_va = loader(Xtr, ytr, True), loader(Xva, yva, False)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

        val_f1 = f1_score(yva, predict(model, dl_va), average="macro")
        sched.step(val_f1)
        print(f"  {tag} ep {ep:02d} val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_state, best_f1


# ---------------- scoring ----------------
def score(y_true, y_pred, n_classes):
    return dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro"),
        recall=recall_score(y_true, y_pred, average=None, labels=np.arange(n_classes)),
    )


def summarize(results, classes):
    """Mean ± sample std (ddof=1) over rounds, then mean per-class recall."""
    accs = np.array([r["acc"] for r in results])
    f1s = np.array([r["f1"] for r in results])
    print(f"\nAccuracy : {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    print(f"Macro-F1 : {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")
    for name, rec in zip(classes, np.mean([r["recall"] for r in results], axis=0)):
        print(f"  {name:6s} recall {rec:.4f}")
