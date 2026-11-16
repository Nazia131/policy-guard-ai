"""Measure the rule-based engine on HarMeme-style data (text field only).

    python -m evaluate.rules_eval --data_dir data/hateful_memes
    python -m evaluate.rules_eval --data_dir data/hateful_memes --show-errors 25
    python -m evaluate.rules_eval --data_dir data/hateful_memes --save-threshold

Honest protocol: the decision threshold is chosen on the *dev* split only; the
*test* split is scored once with that threshold.  Use ``--show-errors`` to read
dev mistakes and improve ``policy/rules.json`` - never tune rules on test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

from data.preprocess.hateful_memes import load_samples
from rules.rule_engine import RuleEngine


def _metrics(y: List[int], p: List[int], s: List[float]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        auc = float("nan")
    n = len(y)
    return {"accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n}


def _score(engine: RuleEngine, data_dir: str, split: str) -> List[Tuple[dict, float]]:
    return [(s, engine.analyze(s["text"])["score"]) for s in load_samples(split, data_dir) if s["label"] in (0, 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data_dir", default=os.path.join("data", "hateful_memes"))
    ap.add_argument("--metric", choices=("accuracy", "f1"), default="accuracy", help="criterion for choosing the dev threshold")
    ap.add_argument("--show-errors", type=int, default=0, help="print N dev false positives and N false negatives")
    ap.add_argument("--save-threshold", action="store_true", help="write the dev-chosen threshold to policy/rules_threshold.json")
    args = ap.parse_args(argv)

    eng = RuleEngine()
    dev = _score(eng, args.data_dir, "dev")
    test = _score(eng, args.data_dir, "test")
    if not dev or not test:
        print("[CHEM][rules_eval] need labelled dev and test splits"); return 1

    yd = [s["label"] for s, _ in dev]
    sd = [x for _, x in dev]
    best = (-1.0, 0.5)
    for i in range(5, 96):
        t = i / 100
        m = _metrics(yd, [int(x >= t) for x in sd], sd)
        if m[args.metric] > best[0]:
            best = (m[args.metric], t)
    thr = best[1]
    print(f"[CHEM][rules_eval] threshold chosen on DEV ({args.metric}): {thr:.2f}  (dev {args.metric}={best[0]:.4f})")

    yt = [s["label"] for s, _ in test]
    st = [x for _, x in test]
    mt = _metrics(yt, [int(x >= thr) for x in st], st)
    base = max(yt.count(0), yt.count(1)) / len(yt)
    print("[CHEM][rules_eval] ===== TEST METRICS (rules, text only) =====")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"[CHEM][rules_eval] {k:9s}: {mt[k]:.4f}")
    print(f"[CHEM][rules_eval] confusion  : TP={mt['tp']} FP={mt['fp']} FN={mt['fn']} TN={mt['tn']} (n={mt['n']})")
    print(f"[CHEM][rules_eval] majority-class baseline accuracy = {base:.4f}")

    if args.show_errors:
        fps = sorted([(x, s) for s, x in dev if s["label"] == 0 and x >= thr], key=lambda t: -t[0])[: args.show_errors]
        fns = sorted([(x, s) for s, x in dev if s["label"] == 1 and x < thr], key=lambda t: t[0])[: args.show_errors]
        print("\n--- DEV false positives (safe, flagged) ---")
        for x, s in fps:
            print(f"{x:.2f}  {s['text'][:140]}")
        print("\n--- DEV false negatives (harmful, missed) ---")
        for x, s in fns:
            print(f"{x:.2f}  {s['text'][:140]}")

    if args.save_threshold:
        path = os.path.join("policy", "rules_threshold.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"harmful_threshold": thr, "chosen_on": "dev", "metric": args.metric}, fh)
        print(f"[CHEM][rules_eval] saved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
