"""Train + evaluate the CLIP image+text linear probe on your meme dataset.

    python -m evaluate.clip_linear_train --data_dir data\\hateful_memes

Embeddings are cached (``--cache``) so re-runs take seconds.  C and the decision
threshold are chosen on DEV; TEST is scored once.  Saves
``model/checkpoints/clip_linear.npz`` for the app.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from data.preprocess.hateful_memes import load_samples
from model.clip_linear import DEFAULT_PATH, clip_embed, extract, featurize


def _acc_auc(y, p, t):
    from sklearn.metrics import roc_auc_score

    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "auc": float(roc_auc_score(y, p)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fit_and_report(tr, dv, te, out_path=None, metric="accuracy"):
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = featurize(tr[0], tr[1]), tr[2]
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda a: (featurize(a[0], a[1]) - mean) / std  # noqa: E731
    Xtr, Xdv, Xte = z(tr), z(dv), z(te)
    best = None
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        p = clf.predict_proba(Xdv)[:, 1]
        for t in np.arange(0.30, 0.71, 0.01):
            m = _acc_auc(dv[2], p, t)
            key = (m[metric], m["auc"])
            if best is None or key > best[0]:
                best = (key, C, float(t), clf)
        print(f"[CHEM][clip_linear] C={C}: dev AUC={_acc_auc(dv[2], p, 0.5)['auc']:.4f}")
    _, C, t, clf = best
    mt = _acc_auc(te[2], clf.predict_proba(Xte)[:, 1], t)
    print(f"[CHEM][clip_linear] chosen on DEV: C={C} threshold={t:.2f}")
    print("[CHEM][clip_linear] ===== TEST METRICS (CLIP image+text linear probe) =====")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"[CHEM][clip_linear] {k:9s}: {mt[k]:.4f}")
    print(f"[CHEM][clip_linear] confusion  : TP={mt['tp']} FP={mt['fp']} FN={mt['fn']} TN={mt['tn']}")
    print(f"[CHEM][clip_linear] majority-class baseline accuracy = {max((te[2] == 0).mean(), (te[2] == 1).mean()):.4f}")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path, coef=clf.coef_.reshape(-1), intercept=clf.intercept_[0], mean=mean, std=std, threshold=t)
        print(f"[CHEM][clip_linear] saved -> {out_path}")
    return mt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data_dir", default=os.path.join("data", "hateful_memes"))
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--cache", default=os.path.join("data", "clip_features.npz"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    if os.path.isfile(args.cache):
        z = np.load(args.cache)
        sets = {s: (z[f"{s}_i"], z[f"{s}_t"], z[f"{s}_y"]) for s in ("train", "dev", "test")}
        print(f"[CHEM][clip_linear] loaded cached embeddings from {args.cache}")
    else:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, pre = clip.load("ViT-B/32", device=device)
        model.eval()
        emb = lambda ims, txts: clip_embed(model, pre, device, ims, txts)  # noqa: E731
        sets = {}
        for s in ("train", "dev", "test"):
            samples = [x for x in load_samples(s, args.data_dir) if x["label"] in (0, 1)]
            sets[s] = extract(samples, emb, args.batch)
        np.savez(args.cache, **{f"{s}_{k}": v for s, (i, t, y) in sets.items() for k, v in (("i", i), ("t", t), ("y", y))})
        print(f"[CHEM][clip_linear] cached embeddings -> {args.cache}")
    fit_and_report(sets["train"], sets["dev"], sets["test"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
