"""Random Forest on hand-crafted features from the shared CWRU artifacts.
Run from the project root: python -m src.train.randomforest"""
from itertools import product

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODELS_DIR, N_LOADS
from src.features import extract
from src.preprocess import load_round, signature
from src.train.common import N_SEEDS, run_seed, save_history, score, summarize

OUT_DIR = MODELS_DIR / "rf"
GRID = dict(max_depth=[None, 20], min_samples_leaf=[1, 5])   # tuned on validation only
N_TREES = 300


def run_round(k, s, d, meta, feats):
    n_classes = len(meta["le"].classes_)
    Ftr, Fva, Fte = feats

    # pick hyperparameters on the validation load only; keep the winning fit
    best_f1, best_params, rf = -1.0, None, None
    for depth, leaf in product(GRID["max_depth"], GRID["min_samples_leaf"]):
        cand = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
            class_weight="balanced", n_jobs=-1, random_state=run_seed(k, s), oob_score=True,
        ).fit(Ftr, d["ytr"])
        f1 = f1_score(d["yva"], cand.predict(Fva), average="macro")
        print(f"  round {k} seed {s} depth={depth} leaf={leaf} val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_params, rf = f1, dict(max_depth=depth, min_samples_leaf=leaf), cand

    # score the test load once
    pred = rf.predict(Fte)
    res = score(d["yte"], pred, n_classes)
    res["importance"] = rf.feature_importances_

    if s == 0:                                         # the app uses the seed-0 checkpoint
        joblib.dump(dict(model=rf, prep=signature(meta)), OUT_DIR / f"model_test{k}.joblib")
    np.savez(OUT_DIR / f"pred_test{k}_seed{s}.npz", y_true=d["yte"], y_pred=pred)
    gap = dict(train_acc=rf.score(Ftr, d["ytr"]), oob_acc=rf.oob_score_,
               val_acc=accuracy_score(d["yva"], rf.predict(Fva)), test_acc=res["acc"])
    save_history(OUT_DIR / f"history_test{k}_seed{s}.json", params=best_params,
                 best_val_f1=best_f1, test_f1=res["f1"], **gap)
    print(f"round {k} seed {s}: {best_params} | train {gap['train_acc']:.4f} | "
          f"OOB {gap['oob_acc']:.4f} | val {gap['val_acc']:.4f} | test {res['acc']:.4f}")
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for k in range(N_LOADS):
        d, meta = load_round(k)
        Ftr, names = extract(d["Wtr"])
        feats = (Ftr, extract(d["Wva"])[0], extract(d["Wte"])[0])
        results.append([run_round(k, s, d, meta, feats) for s in range(N_SEEDS)])
    summarize(results, meta["le"].classes_)

    imp = np.mean([r["importance"] for rk in results for r in rk], axis=0)
    print("\nTop 10 features (mean importance over rounds and seeds):")
    for i in np.argsort(imp)[::-1][:10]:
        print(f"  {names[i]:18s} {imp[i]:.4f}")


if __name__ == "__main__":
    main()
