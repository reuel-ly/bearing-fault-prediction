"""2D-CNN on log-spectrograms of the shared CWRU artifacts.
Run from the project root: python -m src.train.cnn2d"""
import numpy as np
import torch

from src.config import MODELS_DIR, N_LOADS, SEED
from src.models import CNN2D, to_logspec
from src.preprocess import load_round, signature
from src.train.common import loader, predict, score, seed_all, summarize, train_cnn

OUT_DIR = MODELS_DIR / "cnn2d"


def run_round(k):
    seed_all(SEED + k)
    d, meta = load_round(k)
    n_classes = len(meta["le"].classes_)

    # spectrograms, then a second normalization with TRAIN statistics only
    Str, Sva, Ste = to_logspec(d["Wtr"]), to_logspec(d["Wva"]), to_logspec(d["Wte"])
    mu2, sd2 = float(Str.mean()), float(Str.std())
    Str, Sva, Ste = [((S - mu2) / sd2).astype(np.float32) for S in (Str, Sva, Ste)]

    model = CNN2D(n_classes)
    best_state, best_f1 = train_cnn(model, Str, d["ytr"], Sva, d["yva"], n_classes,
                                    tag=f"round {k}")

    # test is touched once, with the best-validation weights
    model.load_state_dict(best_state)
    pred = predict(model, loader(Ste, d["yte"], False))
    res = score(d["yte"], pred, n_classes)

    torch.save(dict(state_dict=best_state, prep=signature(meta), mu2=mu2, sd2=sd2),
               OUT_DIR / f"model_test{k}.pt")
    np.savez(OUT_DIR / f"pred_test{k}.npz", y_true=d["yte"], y_pred=pred)
    print(f"round {k}: val F1 {best_f1:.4f} | test acc {res['acc']:.4f} | test macro-F1 {res['f1']:.4f}")
    return res, meta["le"].classes_


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = [run_round(k) for k in range(N_LOADS)]
    summarize([r for r, _ in out], out[0][1])


if __name__ == "__main__":
    main()
