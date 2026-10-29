"""2D-CNN on log-spectrograms of the shared CWRU artifacts.
Run from the project root: python -m src.train.cnn2d"""
import numpy as np
import torch

from src.config import MODELS_DIR, N_LOADS
from src.models import CNN2D, to_logspec
from src.preprocess import load_round, signature
from src.train.common import (N_SEEDS, loader, predict, run_seed, save_history, score, seed_all,
                              summarize, train_cnn)

OUT_DIR = MODELS_DIR / "cnn2d"


def spectrograms(d):
    """Log-spectrograms, then a second normalization with TRAIN statistics only."""
    Str, Sva, Ste = to_logspec(d["Wtr"]), to_logspec(d["Wva"]), to_logspec(d["Wte"])
    mu2, sd2 = float(Str.mean()), float(Str.std())
    Str, Sva, Ste = [((S - mu2) / sd2).astype(np.float32) for S in (Str, Sva, Ste)]
    return Str, Sva, Ste, mu2, sd2


def run_round(k, s, d, meta, specs):
    seed_all(run_seed(k, s))
    n_classes = len(meta["le"].classes_)
    Str, Sva, Ste, mu2, sd2 = specs

    model = CNN2D(n_classes)
    best_state, best_f1, hist = train_cnn(model, Str, d["ytr"], Sva, d["yva"], n_classes,
                                          tag=f"round {k} seed {s}")

    # test is touched once, with the best-validation weights
    model.load_state_dict(best_state)
    pred = predict(model, loader(Ste, d["yte"], False))
    res = score(d["yte"], pred, n_classes)

    if s == 0:                                         # the app uses the seed-0 checkpoint
        torch.save(dict(state_dict=best_state, prep=signature(meta), mu2=mu2, sd2=sd2),
                   OUT_DIR / f"model_test{k}.pt")
    np.savez(OUT_DIR / f"pred_test{k}_seed{s}.npz", y_true=d["yte"], y_pred=pred)
    save_history(OUT_DIR / f"history_test{k}_seed{s}.json", epochs=hist, best_val_f1=best_f1,
                 test_acc=res["acc"], test_f1=res["f1"])
    print(f"round {k} seed {s}: val F1 {best_f1:.4f} | test acc {res['acc']:.4f} | "
          f"test macro-F1 {res['f1']:.4f}")
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for k in range(N_LOADS):
        d, meta = load_round(k)
        specs = spectrograms(d)
        results.append([run_round(k, s, d, meta, specs) for s in range(N_SEEDS)])
    summarize(results, meta["le"].classes_)


if __name__ == "__main__":
    main()
