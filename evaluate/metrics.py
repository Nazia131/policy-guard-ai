"""Classification metrics for CHEM harm detection.

This module centralises every evaluation metric used across CHEM so that the
training loop, the zero-shot generalisation study and the ablation study all
report numbers computed in exactly the same way.  Nothing here hardcodes harm
semantics: the decision threshold defaults to the ``final_harm_threshold``
coefficient in ``policy/policy_config.json`` and per-class metrics iterate over
the harm categories declared in that same config.

Public API
----------
compute_metrics(y_true, y_prob, threshold=None) -> dict
    Accuracy, precision, recall, F1 and AUC for binary harm detection.
per_class_metrics(y_true, y_pred, categories) -> dict
    Precision / recall / F1 for each harm category (multi-class labels).
pretty_print(metrics) -> str
    Render a metrics dict as an aligned, human-readable table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.config import get_coefficient, get_harm_categories


def _default_threshold() -> float:
    """Return the configured final harm decision threshold.

    Reads ``coefficients.final_harm_threshold`` from ``policy_config.json`` so
    the binarisation cut-off is never hardcoded in Python.  Falls back to
    ``0.5`` only if the coefficient cannot be read (graceful degradation).

    Returns
    -------
    float
        The probability threshold above which a sample is labelled harmful.
    """
    try:
        return float(get_coefficient("final_harm_threshold"))
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[CHEM][metrics] Could not read final_harm_threshold ({exc}); using 0.5")
        return 0.5


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute binary harm-detection metrics from probabilities.

    Parameters
    ----------
    y_true : sequence of int
        Ground-truth binary labels (``1`` = harmful, ``0`` = safe).
    y_prob : sequence of float
        Predicted probability of the *harmful* class in ``[0, 1]``.  Hard
        ``0/1`` predictions are also accepted (they simply threshold trivially).
    threshold : float, optional
        Decision threshold applied to ``y_prob`` to obtain hard predictions.
        Defaults to ``coefficients.final_harm_threshold`` from the policy config.

    Returns
    -------
    dict
        ``{"accuracy", "precision", "recall", "f1", "auc", "threshold",
        "n_samples"}``.  ``auc`` is ``float('nan')`` when only one class is
        present in ``y_true`` (AUC is undefined in that case).
    """
    thr = _default_threshold() if threshold is None else float(threshold)

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)

    if y_true_arr.size == 0:
        # No data to score: return a well-formed, all-zero record.
        print("[CHEM][metrics] compute_metrics received no samples; returning zeros")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
            "threshold": thr,
            "n_samples": 0,
        }

    y_pred = (y_prob_arr >= thr).astype(int)

    # AUC needs both classes present and continuous scores; guard against the
    # single-class edge case which is common when a whole category is held out.
    try:
        if len(np.unique(y_true_arr)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[CHEM][metrics] AUC could not be computed ({exc}); reporting NaN")
        auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "auc": auc,
        "threshold": thr,
        "n_samples": int(y_true_arr.size),
    }
    return metrics


def per_class_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    categories: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision / recall / F1 for each harm category.

    Unlike :func:`compute_metrics` (which scores the binary harmful-vs-safe
    decision) this treats the problem as multi-class over harm-category ids so
    per-category strengths and weaknesses become visible.

    Parameters
    ----------
    y_true : sequence
        Ground-truth harm-category id per sample (e.g. ``"hate_speech"``).
        Integer class indices aligned to ``categories`` are also accepted.
    y_pred : sequence
        Predicted harm-category id per sample, in the same label space.
    categories : sequence of str, optional
        The ordered set of harm-category ids to report.  Defaults to the keys
        of ``harm_categories`` in ``policy_config.json``.

    Returns
    -------
    dict
        ``{category_id: {"precision", "recall", "f1", "support"}}`` plus a
        ``"macro_avg"`` aggregate over the categories.
    """
    if categories is None:
        try:
            categories = list(get_harm_categories().keys())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[CHEM][metrics] Could not read harm categories ({exc})")
            categories = sorted({*map(str, y_true), *map(str, y_pred)})
    labels = list(categories)

    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    if not y_true_list:
        print("[CHEM][metrics] per_class_metrics received no samples")
        return {"macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}}

    # Per-label precision/recall/f1/support with the label universe pinned to the
    # configured categories so absent categories still appear (as zeros).
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_list,
        y_pred_list,
        labels=labels,
        zero_division=0,
    )

    result: Dict[str, Dict[str, float]] = {}
    for i, category in enumerate(labels):
        result[category] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Macro average gives every category equal weight regardless of frequency.
    result["macro_avg"] = {
        "precision": float(np.mean(precision)) if len(precision) else 0.0,
        "recall": float(np.mean(recall)) if len(recall) else 0.0,
        "f1": float(np.mean(f1)) if len(f1) else 0.0,
        "support": int(np.sum(support)) if len(support) else 0,
    }
    return result


def pretty_print(metrics: Dict[str, Any]) -> str:
    """Render a metrics dict as an aligned, human-readable table.

    Handles both the flat binary dict returned by :func:`compute_metrics` and
    the nested per-category dict returned by :func:`per_class_metrics`.

    Parameters
    ----------
    metrics : dict
        A metrics mapping produced by this module.

    Returns
    -------
    str
        The formatted table.  It is also printed to stdout as a side effect so
        callers can use it purely for logging.
    """
    lines: List[str] = []
    is_nested = any(isinstance(v, dict) for v in metrics.values())

    if is_nested:
        header = f"{'category':<26}{'precision':>11}{'recall':>10}{'f1':>8}{'support':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, row in metrics.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"{name:<26}"
                f"{row.get('precision', 0.0):>11.4f}"
                f"{row.get('recall', 0.0):>10.4f}"
                f"{row.get('f1', 0.0):>8.4f}"
                f"{int(row.get('support', 0)):>10d}"
            )
    else:
        lines.append(f"{'metric':<14}{'value':>12}")
        lines.append("-" * 26)
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:<14}{value:>12.4f}")
            else:
                lines.append(f"{key:<14}{str(value):>12}")

    table = "\n".join(lines)
    print(table)
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluate a trained GAT checkpoint on the held-out test split.

    Mirrors the split used by ``model.train`` (70/15/15, seed 42) so the
    reported numbers are on the same test set as the training run.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv``).

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on handled failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a CHEM GAT checkpoint on the held-out test split.",
    )
    parser.add_argument(
        "--checkpoint",
        default="model/checkpoints/best_gat.pt",
        help="Path to the saved GAT checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Hateful Memes dataset root (default: data/hateful_memes).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: config final_harm_threshold).",
    )
    args = parser.parse_args(argv)

    try:
        import torch
        from torch_geometric.loader import DataLoader

        from encoders.image_encoder import ImageEncoder
        from encoders.text_encoder import TextEncoder
        from model.gat_model import CHEMGraphAttentionNetwork
        from model.train import _build_dataset, _load_samples, _split_dataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CHEMGraphAttentionNetwork().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.eval()

        samples = _load_samples("train", args.data_dir) + _load_samples(
            "dev", args.data_dir
        )
        graphs = _build_dataset(
            samples, TextEncoder(device=str(device)), ImageEncoder(device=str(device))
        )
        test_graphs = _split_dataset(graphs)["test"]
        if not test_graphs:
            print("[CHEM][metrics] No test graphs available; nothing to evaluate.")
            return 1

        y_true: List[int] = []
        y_prob: List[float] = []
        with torch.no_grad():
            for batch in DataLoader(test_graphs, batch_size=32, shuffle=False):
                batch = batch.to(device)
                log_probs = model(batch.x.float(), batch.edge_index, batch.batch)
                y_prob.extend(torch.exp(log_probs)[:, 1].cpu().tolist())
                y_true.extend(batch.y.view(-1).long().cpu().tolist())
    except Exception as exc:
        print(f"[CHEM][metrics] Evaluation failed: {exc}")
        return 1

    print(f"[CHEM][metrics] Test set: {len(y_true)} samples")
    pretty_print(compute_metrics(y_true, y_prob, threshold=args.threshold))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
