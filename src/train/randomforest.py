"""Random Forest on hand-crafted features from the shared CWRU artifacts.
Run from the project root: python -m src.train.randomforest"""
from itertools import product

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.config import MODELS_DIR, N_LOADS, SEED
from src.features import extract
from src.preprocess import load_round, signature
from src.train.common import score, summarize

OUT_DIR = MODELS_DIR / "rf"
GRID = dict(max_depth=[None, 20], min_samples_leaf=[1, 5])   # tuned on validation only
N_TREES = 300


def run_round(k):
    d, meta = load_round(k)
    n_classes = len(meta["le"].classes_)

    Ftr, names = extract(d["Wtr"])
    Fva, _ = extract(d["Wva"])
    Fte, _ = extract(d["Wte"])

    # pick hyperparameters on the validation load only; keep the winning fit
    best_f1, best_params, rf = -1.0, None, None
    for depth, leaf in product(GRID["max_depth"], GRID["min_samples_leaf"]):
        cand = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
            class_weight="balanced", n_jobs=-1, random_state=SEED + k,
        ).fit(Ftr, d["ytr"])
        f1 = f1_score(d["yva"], cand.predict(Fva), average="macro")
        print(f"  round {k} depth={depth} leaf={leaf} val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_params, rf = f1, dict(max_depth=depth, min_samples_leaf=leaf), cand

    # score the test load once
    pred = rf.predict(Fte)
    res = score(d["yte"], pred, n_classes)
    res["importance"] = rf.feature_importances_

    joblib.dump(dict(model=rf, prep=signature(meta)), OUT_DIR / f"model_test{k}.joblib")
    np.savez(OUT_DIR / f"pred_test{k}.npz", y_true=d["yte"], y_pred=pred)
    print(f"round {k}: {best_params} | val F1 {best_f1:.4f} | "
          f"test acc {res['acc']:.4f} | test macro-F1 {res['f1']:.4f}")
    return res, names, meta["le"].classes_


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = [run_round(k) for k in range(N_LOADS)]
    results, names = [r for r, _, _ in out], out[0][1]
    summarize(results, out[0][2])

    imp = np.mean([r["importance"] for r in results], axis=0)
    print("\nTop 10 features (mean importance over rounds):")
    for i in np.argsort(imp)[::-1][:10]:
        print(f"  {names[i]:18s} {imp[i]:.4f}")


if __name__ == "__main__":
    main()
