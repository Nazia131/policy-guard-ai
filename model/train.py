"""CHEM GAT training pipeline (Facebook Hateful Memes).

End-to-end training of :class:`model.gat_model.CHEMGraphAttentionNetwork` on the
Facebook Hateful Memes dataset.  Each meme (image + overlaid text + binary
label) is turned into a *Content Composition Graph* and the GAT learns to
classify the whole graph as SAFE / HARMFUL.

Pipeline
--------
1. **Load** Hateful Memes samples via ``data.preprocess.hateful_memes``.
2. **Encode** each sample with the CHEM text + image encoders (loaded once).
3. **Build** a PyG ``Data`` graph per sample via ``graph.graph_builder.build_graph``.
4. **Split** 70 / 15 / 15 into train / val / test and batch with PyG ``DataLoader``.
5. **Train** with ``NLLLoss`` on the model's ``log_softmax`` output (equivalent to
   BCE for the 2-class case), Adam (``lr=0.005``, ``weight_decay=5e-4``), with
   early stopping (patience 20) on validation F1; the best checkpoint is saved to
   ``model/checkpoints/best_gat.pt``.
6. **Log** every metric to Weights & Biases (guarded so training still runs if
   W&B is offline / unavailable).
7. **Evaluate** on the held-out test set, printing accuracy / precision / recall
   / F1 / AUC.

Expected data interface
------------------------
This module targets the documented interface
``data.preprocess.hateful_memes.load_hateful_memes(split) -> iterable of
{"id", "img", "text", "label"}``.  If the concrete module exposes a differently
named loader, adapt :func:`_load_samples` -- the rest of the pipeline is agnostic
to the loader's internals.

Run
---
``python -m model.train --epochs 100 --batch_size 32 --data_dir data/hateful_memes``
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from model.gat_model import CHEMGraphAttentionNetwork

# Default checkpoint location (created on demand).
_CHECKPOINT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "checkpoints"))
_BEST_CHECKPOINT_PATH = os.path.join(_CHECKPOINT_DIR, "best_gat.pt")

# Training hyper-parameters fixed by the CHEM spec / INTERFACES.md.
_LEARNING_RATE = 0.005
_WEIGHT_DECAY = 5e-4
_EARLY_STOP_PATIENCE = 20

# Anti-collapse settings.
_GRAD_CLIP_NORM = 1.0          # stops one noisy batch from flipping the whole model
_COLLAPSE_RATE = 0.05          # predicting one class for >95% of graphs = collapsed
_SELECT_CHOICES = ("auc", "balanced_accuracy", "macro_f1", "f1")
_DEFAULT_SELECT_BY = "auc"     # NOT f1: F1 rewards "everything is harmful"


# --------------------------------------------------------------------------- #
# Data loading + graph construction
# --------------------------------------------------------------------------- #
def _load_samples(split: str, data_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Load raw Hateful Memes samples for a split via the preprocess module.

    Parameters
    ----------
    split : str
        Dataset split name (``"train"`` / ``"dev"`` / ``"test"`` etc.).
    data_dir : str, optional
        Optional dataset root forwarded to the loader if it accepts one.

    Returns
    -------
    list[dict]
        Samples shaped ``{"id", "img", "text", "label"}``.  Returns an empty
        list (with a warning) if the loader is unavailable, so callers can
        degrade gracefully.
    """
    try:
        from data.preprocess import hateful_memes
    except Exception as exc:  # pragma: no cover - preprocess may not be present
        print(f"[CHEM][train] WARNING: could not import hateful_memes loader: {exc}")
        return []

    # Prefer the documented `load_hateful_memes(split)` interface; fall back to
    # any single obvious loader callable the module exposes.
    loader = getattr(hateful_memes, "load_hateful_memes", None)
    if loader is None:
        for name in ("load", "load_split", "iter_samples"):
            loader = getattr(hateful_memes, name, None)
            if loader is not None:
                break
    if loader is None:
        print(
            "[CHEM][train] WARNING: hateful_memes exposes no known loader "
            "(expected load_hateful_memes(split))."
        )
        return []

    try:
        if data_dir is None:
            # Fall back to the CHEM_DATA_DIR env var / data/hateful_memes default.
            samples = hateful_memes.load_samples(split)
        else:
            # load_hateful_memes(data_dir, split) — data_dir first, then split name.
            samples = loader(data_dir, split)  # type: ignore[misc]
        return list(samples)
    except Exception as exc:  # pragma: no cover
        print(f"[CHEM][train] WARNING: loader failed for split={split!r}: {exc}")
        return []


def _build_dataset(
    samples: List[Dict[str, Any]],
    text_encoder,
    image_encoder,
) -> List[Any]:
    """Encode + graph-build every sample into a list of PyG ``Data`` objects.

    Parameters
    ----------
    samples : list[dict]
        Raw ``{"id","img","text","label"}`` samples.
    text_encoder : encoders.text_encoder.TextEncoder
        Loaded text encoder (reused across all samples).
    image_encoder : encoders.image_encoder.ImageEncoder
        Loaded image encoder (reused across all samples).

    Returns
    -------
    list
        ``torch_geometric.data.Data`` graphs carrying node features, edges and a
        ``y`` label.  Samples that fail to encode/build are skipped with a note.
    """
    from graph.graph_builder import build_graph
    from PIL import Image

    dataset: List[Any] = []
    total = len(samples)
    image_failed = 0       # image could not be opened / encoded
    image_no_elements = 0  # image encoded but contributed no graph nodes
    for i, sample in enumerate(samples):
        try:
            text = sample.get("text", "") or ""
            label = int(sample.get("label", 0))

            # --- Text elements. ---------------------------------------------
            text_out = text_encoder.encode(text)
            elements = list(text_out.get("elements", []))

            # --- Image elements (best-effort; skip image on failure). -------
            img_field = sample.get("img")
            if img_field is not None:
                try:
                    if not isinstance(img_field, str):
                        image = img_field
                    else:
                        # The loader already resolves ``img`` to an absolute path.
                        image = Image.open(img_field).convert("RGB")
                    image_out = image_encoder.encode(image)
                    image_elements = list(image_out.get("elements", []))
                    if not image_elements:
                        image_no_elements += 1
                    elements.extend(image_elements)
                except Exception as img_exc:  # pragma: no cover
                    image_failed += 1
                    if image_failed <= 5:  # don't flood the log
                        print(
                            f"[CHEM][train] sample {sample.get('id', i)}: image "
                            f"encode failed ({img_exc}); using text only."
                        )
            else:
                image_failed += 1

            if not elements:
                continue

            # --- Build the Content Composition Graph. -----------------------
            graph_dict = build_graph(elements, label)
            data = graph_dict["data"]
            dataset.append(data)
        except Exception as exc:  # pragma: no cover - never let one bad sample stop training
            print(f"[CHEM][train] skipping sample {sample.get('id', i)}: {exc}")

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[CHEM][train] built graphs {i + 1}/{total}")

    # Hateful-meme harm comes from image + text *together*.  Graphs built from the
    # text alone cannot carry that signal, so a high share of them starves the model
    # and it falls back to the class prior (= "label everything harmful").
    text_only = image_failed + image_no_elements
    if total and text_only / total > 0.05:
        print(
            f"[CHEM][train] WARNING: {text_only}/{total} samples ({text_only / total:.0%}) "
            f"have NO image nodes (failed/missing: {image_failed}, "
            f"encoded-but-empty: {image_no_elements}). Fix image paths / the image "
            "encoder before trusting any accuracy number."
        )
    else:
        print(f"[CHEM][train] image coverage OK ({total - text_only}/{total} graphs include image nodes).")

    return dataset


def _graph_label(graph: Any) -> int:
    """Return the integer class label (0/1) stored on a PyG graph."""
    return int(graph.y.view(-1)[0].item())


def _split_dataset(dataset: List[Any], seed: int = 42) -> Dict[str, List[Any]]:
    """Split a graph list 70 / 15 / 15 into train / val / test, *stratified by label*.

    Every split keeps (almost exactly) the dataset's harmful / safe ratio.  A
    plain random shuffle lets the ratio drift between splits (e.g. 60 % harmful
    in train but 53 % in test), which silently shifts accuracy and makes
    validation-based decisions unreliable on small data.

    ``evaluate/metrics.py`` and the ablation study call this same function, so
    training and evaluation always see the same test set.

    Parameters
    ----------
    dataset : list
        List of PyG ``Data`` graphs (each carrying ``y``).
    seed : int, optional
        RNG seed for a reproducible shuffle (default 42).

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[Any]] = {}
    for graph in dataset:
        by_label.setdefault(_graph_label(graph), []).append(graph)

    train: List[Any] = []
    val: List[Any] = []
    test: List[Any] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train += group[:n_train]
        val += group[n_train : n_train + n_val]
        test += group[n_train + n_val :]

    # Mix the classes back together so batches are not label-sorted.
    for part in (train, val, test):
        rng.shuffle(part)
    return {"train": train, "val": val, "test": test}


def _class_weights(graphs: Sequence[Any]) -> torch.Tensor:
    """Inverse-frequency class weights ``[w_safe, w_harmful]`` for the loss.

    With weights, a mistake on the minority class costs proportionally more, so
    "always predict the majority class" stops being the cheapest solution.
    Falls back to equal weights when a class is missing.
    """
    counts = [0, 0]
    for graph in graphs:
        counts[_graph_label(graph)] += 1
    total = sum(counts)
    if total == 0 or min(counts) == 0:
        return torch.ones(2)
    return torch.tensor([total / (2.0 * counts[0]), total / (2.0 * counts[1])])


def _set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run (init, shuffling, dropout) repeats."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _compute_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, float]:
    """Compute accuracy / precision / recall / F1 / AUC for binary predictions.

    Uses ``evaluate.metrics.compute_metrics`` when available, otherwise falls
    back to scikit-learn directly.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth binary labels.
    y_prob : list[float]
        Predicted harmful-class probabilities.

    Returns
    -------
    dict
        ``{"accuracy","precision","recall","f1","auc"}`` (missing values 0.0).
    """
    try:
        from evaluate.metrics import compute_metrics

        return compute_metrics(y_true, y_prob)
    except Exception:
        pass  # fall through to sklearn

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    except Exception:  # pragma: no cover
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def _prediction_stats(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Collapse-aware statistics for a set of predictions.

    ``accuracy`` and ``f1`` alone cannot tell a real model from a constant one:
    "always harmful" scores accuracy = harmful share and, when harmful is the
    majority, a *high* F1.  These numbers can:

    * ``pred_pos_rate``    share of graphs predicted harmful (near 0 or 1 = collapse)
    * ``base_rate``        true share of harmful graphs
    * ``majority_acc``     accuracy of always guessing the majority class (the bar
      a model must beat to be worth anything)
    * ``balanced_accuracy`` mean of recall on each class (0.5 = no skill)
    * ``macro_f1``         mean of per-class F1 (a constant predictor scores <= 0.5)
    * ``prob_std``         spread of the predicted probabilities (~0 = constant output)
    """
    n = len(y_true)
    if n == 0:
        return {k: 0.0 for k in (
            "pred_pos_rate", "base_rate", "majority_acc", "balanced_accuracy",
            "macro_f1", "prob_std", "accuracy")}
    pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, pred) if t == 1 and p == 0)

    def _div(a: float, b: float) -> float:
        return a / b if b else 0.0

    recall_pos, recall_neg = _div(tp, tp + fn), _div(tn, tn + fp)

    def _f1(p_tp: int, p_fp: int, p_fn: int) -> float:
        prec, rec = _div(p_tp, p_tp + p_fp), _div(p_tp, p_tp + p_fn)
        return _div(2 * prec * rec, prec + rec)

    mean_p = sum(y_prob) / n
    base = sum(y_true) / n
    return {
        "pred_pos_rate": sum(pred) / n,
        "base_rate": base,
        "majority_acc": max(base, 1.0 - base),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": 0.5 * (recall_pos + recall_neg),
        "macro_f1": 0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp)),
        "prob_std": (sum((p - mean_p) ** 2 for p in y_prob) / n) ** 0.5,
    }


def _is_collapsed(stats: Dict[str, float]) -> bool:
    """True when the model predicts (almost) one class for everything."""
    rate = stats.get("pred_pos_rate", 0.5)
    return rate > 1.0 - _COLLAPSE_RATE or rate < _COLLAPSE_RATE or stats.get("prob_std", 1.0) < 0.01


def _best_threshold(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, float]:
    """Pick the decision threshold that maximises balanced accuracy.

    A model can rank well (good AUC) yet look "collapsed" at 0.5 simply because
    its probabilities sit in a narrow band.  Re-centring the threshold on the
    validation set fixes that without retraining.

    Returns
    -------
    (threshold, balanced_accuracy)
    """
    if not y_true or len(set(y_true)) < 2:
        return 0.5, 0.0
    best_thr, best_bal = 0.5, -1.0
    for step in range(5, 96):
        thr = step / 100.0
        bal = _prediction_stats(y_true, y_prob, thr)["balanced_accuracy"]
        # '>' (not '>=') + ascending scan keeps the lowest of tied thresholds; prefer
        # the one closest to 0.5 for stability.
        if bal > best_bal + 1e-12 or (abs(bal - best_bal) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_thr, best_bal = thr, bal
    return best_thr, best_bal


def _selection_score(
    metrics: Dict[str, float], stats: Dict[str, float], select_by: str
) -> float:
    """Score used to pick the best epoch (higher is better).

    Default is validation AUC: it is threshold-free and a constant predictor
    scores exactly 0.5, so it cannot reward collapse the way F1 does.
    """
    if select_by == "auc":
        return float(metrics.get("auc", 0.0))
    if select_by == "balanced_accuracy":
        return float(stats.get("balanced_accuracy", 0.0))
    if select_by == "macro_f1":
        return float(stats.get("macro_f1", 0.0))
    return float(metrics.get("f1", 0.0))  # legacy behaviour


def _evaluate(model, loader, device) -> Dict[str, Any]:
    """Run the model over a loader and return metrics + raw predictions.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The model in any mode (set to eval internally).
    loader : torch_geometric.loader.DataLoader
        Batched graphs.
    device : torch.device
        Compute device.

    Returns
    -------
    dict
        ``{"metrics": {...}, "loss": float}``.
    """
    model.eval()
    criterion = nn.NLLLoss()
    y_true: List[int] = []
    y_prob: List[float] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            log_probs = model(batch.x.float(), batch.edge_index, batch.batch)
            target = batch.y.view(-1).long()
            total_loss += float(criterion(log_probs, target).item())
            n_batches += 1

            probs = torch.exp(log_probs)[:, 1]  # harmful-class probability
            y_prob.extend(probs.cpu().tolist())
            y_true.extend(target.cpu().tolist())

    metrics = _compute_metrics(y_true, y_prob)
    return {
        "metrics": metrics,
        "loss": total_loss / max(n_batches, 1),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def _train_one_epoch(
    model, loader, optimizer, criterion, device, grad_clip: Optional[float] = _GRAD_CLIP_NORM
) -> float:
    """Train the model for a single epoch and return the mean loss.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The model (set to train mode internally).
    loader : torch_geometric.loader.DataLoader
        Batched training graphs.
    optimizer : torch.optim.Optimizer
        The optimiser.
    criterion : torch.nn.Module
        Loss function (``NLLLoss``).
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        log_probs = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(log_probs, batch.y.view(-1).long())
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _fit(
    splits: Dict[str, List[Any]],
    *,
    device: Any = "cpu",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = _LEARNING_RATE,
    weight_decay: float = _WEIGHT_DECAY,
    dropout: Optional[float] = None,
    select_by: str = _DEFAULT_SELECT_BY,
    class_weighted: bool = True,
    grad_clip: Optional[float] = _GRAD_CLIP_NORM,
    seed: int = 42,
    patience: int = _EARLY_STOP_PATIENCE,
    checkpoint_path: str = _BEST_CHECKPOINT_PATH,
    wandb: Any = None,
) -> Dict[str, Any]:
    """Train on ``splits`` with collapse protection and report honest metrics.

    What protects against the "everything is harmful" collapse:

    * class-weighted loss (``class_weighted``) so the majority class is not free,
    * checkpoint / early-stopping selection on validation AUC (``select_by``) so a
      constant predictor can never win,
    * gradient clipping and a fixed seed so one noisy batch cannot flip the model,
    * a per-epoch monitor that prints the share of graphs predicted harmful next
      to the true share, and flags COLLAPSED epochs,
    * a validation-tuned decision threshold stored in the checkpoint, and a final
      report that compares against the "always guess the majority" baseline.

    Returns
    -------
    dict
        ``{"checkpoint", "best_epoch", "threshold", "test_metrics", "test_stats",
        "test_stats_tuned", "learned_signal", "beats_majority"}``.
    """
    from torch_geometric.loader import DataLoader

    if select_by not in _SELECT_CHOICES:
        raise ValueError(f"select_by must be one of {_SELECT_CHOICES}, got {select_by!r}")

    _set_seed(seed)
    device = torch.device(device)
    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)

    model = (
        CHEMGraphAttentionNetwork(dropout=dropout)
        if dropout is not None
        else CHEMGraphAttentionNetwork()
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = _class_weights(splits["train"]).to(device) if class_weighted else None
    criterion = nn.NLLLoss(weight=weights)
    if weights is not None:
        print(
            f"[CHEM][train] class weights [safe, harmful] = "
            f"[{weights[0].item():.2f}, {weights[1].item():.2f}]"
        )
    print(
        f"[CHEM][train] lr={lr} weight_decay={weight_decay} dropout="
        f"{model.dropout} grad_clip={grad_clip} select_by={select_by} seed={seed}"
    )

    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    collapsed_epochs = 0
    print("[CHEM][train] Starting training ...")
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=grad_clip
        )
        val_eval = _evaluate(model, val_loader, device)
        val_metrics = val_eval["metrics"]
        stats = _prediction_stats(val_eval["y_true"], val_eval["y_prob"])
        collapsed = _is_collapsed(stats)
        collapsed_epochs += int(collapsed)
        score = _selection_score(val_metrics, stats, select_by)

        print(
            f"[CHEM][train] epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | val_auc={val_metrics.get('auc', 0.0):.3f} | "
            f"val_bal_acc={stats['balanced_accuracy']:.3f} | val_acc={stats['accuracy']:.3f} | "
            f"predicted_harmful={stats['pred_pos_rate']:.0%} (true {stats['base_rate']:.0%})"
            + ("  <-- COLLAPSED" if collapsed else "")
        )
        _wandb_log(
            wandb,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_eval["loss"],
                "val_pred_harmful_rate": stats["pred_pos_rate"],
                "val_balanced_accuracy": stats["balanced_accuracy"],
                "val_macro_f1": stats["macro_f1"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            },
        )

        if score > best_score + 1e-4:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            threshold, _ = _best_threshold(val_eval["y_true"], val_eval["y_prob"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "val_f1": float(val_metrics.get("f1", 0.0)),  # kept for compatibility
                    "val_score": best_score,
                    "select_by": select_by,
                    "threshold": threshold,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(
                f"[CHEM][train] new best val_{select_by}={best_score:.4f} "
                f"(threshold {threshold:.2f}); saved -> {checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[CHEM][train] early stopping at epoch {epoch} "
                    f"(no val {select_by} improvement in {patience} epochs)."
                )
                break

    # --- Final test-set evaluation using the best checkpoint. --------------
    print("[CHEM][train] Loading best checkpoint for test evaluation ...")
    threshold = 0.5
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        threshold = float(checkpoint.get("threshold", 0.5))

    test_eval = _evaluate(model, test_loader, device)
    test_metrics = test_eval["metrics"]
    y_true, y_prob = test_eval["y_true"], test_eval["y_prob"]
    stats = _prediction_stats(y_true, y_prob)
    stats_tuned = _prediction_stats(y_true, y_prob, threshold)
    auc = float(test_metrics.get("auc", 0.0))

    print("[CHEM][train] ===== TEST METRICS (threshold 0.50) =====")
    print(f"[CHEM][train] accuracy          : {stats['accuracy']:.4f}")
    print(f"[CHEM][train] balanced accuracy : {stats['balanced_accuracy']:.4f}   (0.50 = no skill)")
    print(f"[CHEM][train] macro F1          : {stats['macro_f1']:.4f}")
    print(f"[CHEM][train] precision/recall  : {test_metrics.get('precision', 0.0):.4f} / {test_metrics.get('recall', 0.0):.4f}")
    print(f"[CHEM][train] AUC               : {auc:.4f}   (0.50 = no skill)")
    print(f"[CHEM][train] predicted harmful : {stats['pred_pos_rate']:.0%}   (true share {stats['base_rate']:.0%})")
    print(f"[CHEM][train] always-guess-majority accuracy = {stats['majority_acc']:.4f}")
    print(
        f"[CHEM][train] at tuned threshold {threshold:.2f}: accuracy={stats_tuned['accuracy']:.4f} "
        f"balanced_acc={stats_tuned['balanced_accuracy']:.4f} "
        f"predicted_harmful={stats_tuned['pred_pos_rate']:.0%}"
    )

    learned_signal = auc >= 0.55
    beats_majority = stats["accuracy"] > stats["majority_acc"] or stats_tuned["balanced_accuracy"] > 0.55
    if not learned_signal:
        print(
            "[CHEM][train] WARNING: test AUC is ~0.5 - the model has NOT learned a usable "
            "signal. Any accuracy above is just the class ratio, not skill. Check the "
            "image-encode failures above, the labels, and the features before tuning more."
        )
    elif _is_collapsed(stats) and not _is_collapsed(stats_tuned):
        print(
            "[CHEM][train] NOTE: the model ranks well but is biased at 0.50; use the tuned "
            f"threshold {threshold:.2f} stored in the checkpoint."
        )
    if collapsed_epochs:
        print(f"[CHEM][train] {collapsed_epochs} epoch(s) were collapsed to one class.")

    _wandb_log(wandb, {f"test_{k}": v for k, v in test_metrics.items()})
    return {
        "checkpoint": checkpoint_path,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "test_metrics": test_metrics,
        "test_stats": stats,
        "test_stats_tuned": stats_tuned,
        "learned_signal": bool(learned_signal),
        "beats_majority": bool(beats_majority),
    }


def train(
    epochs: int = 100,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    wandb_project: str = "chem",
    lr: float = _LEARNING_RATE,
    weight_decay: float = _WEIGHT_DECAY,
    dropout: Optional[float] = None,
    select_by: str = _DEFAULT_SELECT_BY,
    class_weighted: bool = True,
    seed: int = 42,
) -> Optional[str]:
    """Run the full CHEM GAT training + evaluation pipeline.

    Parameters
    ----------
    epochs : int, optional
        Maximum number of epochs (early stopping may end sooner). Default 100.
    batch_size : int, optional
        Graphs per mini-batch (default 32).
    data_dir : str, optional
        Dataset root forwarded to the Hateful Memes loader.
    wandb_project : str, optional
        Weights & Biases project name (default ``"chem"``).
    lr, weight_decay : float, optional
        Adam learning rate / L2 penalty.
    dropout : float, optional
        GAT dropout; ``None`` keeps the architecture default (0.6).
    select_by : str, optional
        Metric that picks the best epoch: ``auc`` (default), ``balanced_accuracy``,
        ``macro_f1`` or the legacy ``f1`` (which rewards predicting everything harmful).
    class_weighted : bool, optional
        Weight the loss by inverse class frequency (default True).
    seed : int, optional
        Seed for init, shuffling and dropout (default 42).

    Returns
    -------
    str or None
        Path to the best saved checkpoint, or ``None`` if no data was available
        to train on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CHEM][train] Using device: {device}")

    # --- Weights & Biases (guarded so offline / missing wandb never crashes). -
    wandb = _init_wandb(wandb_project, epochs, batch_size)

    # --- Encoders loaded ONCE and reused for every sample. ------------------
    print("[CHEM][train] Loading text + image encoders (once) ...")
    from encoders.text_encoder import TextEncoder
    from encoders.image_encoder import ImageEncoder

    text_encoder = TextEncoder(device=str(device))
    image_encoder = ImageEncoder(device=str(device))

    # --- Build the dataset. -------------------------------------------------
    print("[CHEM][train] Loading Hateful Memes samples ...")
    samples = _load_samples("train", data_dir)
    # Some distributions ship a separate dev split; fold it in when present.
    samples += _load_samples("dev", data_dir)
    print(f"[CHEM][train] Loaded {len(samples)} raw samples.")

    # Blind-test rows carry label -1 and anything else is unusable: drop them
    # loudly instead of crashing (or worse, training on a wrong target).
    labelled = [s for s in samples if int(s.get("label", -1)) in (0, 1)]
    if len(labelled) != len(samples):
        print(
            f"[CHEM][train] WARNING: dropped {len(samples) - len(labelled)} samples "
            "without a 0/1 label."
        )
    samples = labelled

    if not samples:
        print("[CHEM][train] ERROR: no samples available; aborting training.")
        return None

    harmful = sum(int(s["label"]) for s in samples)
    print(
        f"[CHEM][train] Label balance: {harmful} harmful / {len(samples) - harmful} safe "
        f"({harmful / len(samples):.0%} harmful)"
    )

    dataset = _build_dataset(samples, text_encoder, image_encoder)
    print(f"[CHEM][train] Built {len(dataset)} content-composition graphs.")
    if not dataset:
        print("[CHEM][train] ERROR: no graphs built; aborting training.")
        return None

    splits = _split_dataset(dataset)
    for name, part in splits.items():
        pos = sum(_graph_label(g) for g in part)
        print(
            f"[CHEM][train] {name:5s}: {len(part)} graphs, {pos / max(len(part), 1):.0%} harmful"
        )

    result = _fit(
        splits,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        select_by=select_by,
        class_weighted=class_weighted,
        seed=seed,
        wandb=wandb,
    )
    _wandb_finish(wandb)
    return result["checkpoint"]


# --------------------------------------------------------------------------- #
# Weights & Biases helpers (graceful degradation)
# --------------------------------------------------------------------------- #
def _init_wandb(project: str, epochs: int, batch_size: int):
    """Initialise a W&B run, honouring ``WANDB_MODE`` and failing soft.

    Parameters
    ----------
    project : str
        W&B project name.
    epochs, batch_size : int
        Hyper-parameters recorded in the run config.

    Returns
    -------
    module or None
        The ``wandb`` module if a run started, else ``None`` (training then runs
        without logging).
    """
    try:
        import wandb

        mode = os.environ.get("WANDB_MODE", "online")
        wandb.init(
            project=project,
            mode=mode,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": _LEARNING_RATE,
                "weight_decay": _WEIGHT_DECAY,
                "early_stop_patience": _EARLY_STOP_PATIENCE,
                "architecture": "GAT(768->256x8->2)",
            },
        )
        print(f"[CHEM][train] Weights & Biases initialised (mode={mode}).")
        return wandb
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"[CHEM][train] W&B unavailable ({exc}); continuing without logging.")
        return None


def _wandb_log(wandb, payload: Dict[str, Any]) -> None:
    """Log a metric dict to W&B if a run is active (no-op otherwise).

    Parameters
    ----------
    wandb : module or None
        The active wandb module, or None.
    payload : dict
        Metrics to log.

    Returns
    -------
    None
    """
    if wandb is None:
        return
    try:
        wandb.log(payload)
    except Exception:  # pragma: no cover
        pass


def _wandb_finish(wandb) -> None:
    """Finish the active W&B run if any (no-op otherwise).

    Parameters
    ----------
    wandb : module or None
        The active wandb module, or None.

    Returns
    -------
    None
    """
    if wandb is None:
        return
    try:
        wandb.finish()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Parse CLI arguments and launch training.

    Inputs: command-line arguments (``--epochs``, ``--batch_size``,
    ``--data_dir``, ``--wandb_project``).
    Outputs: none (side effect: trains + saves a checkpoint, prints metrics).
    """
    parser = argparse.ArgumentParser(description="Train the CHEM GAT harm classifier.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=32, help="Graphs per batch.")
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Hateful Memes dataset root (forwarded to the loader).",
    )
    parser.add_argument(
        "--wandb_project",
        "--wandb-project",
        dest="wandb_project",
        type=str,
        default="chem",
        help="Weights & Biases project name.",
    )
    parser.add_argument("--lr", type=float, default=_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--weight_decay", "--weight-decay", dest="weight_decay", type=float,
        default=_WEIGHT_DECAY, help="Adam weight decay.",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="GAT dropout (default: architecture default, 0.6).",
    )
    parser.add_argument(
        "--select_by", "--select-by", dest="select_by", choices=_SELECT_CHOICES,
        default=_DEFAULT_SELECT_BY,
        help="Metric that picks the best epoch. 'f1' is the legacy rule and rewards "
        "predicting everything harmful.",
    )
    parser.add_argument(
        "--no_class_weights", "--no-class-weights", dest="class_weighted",
        action="store_false", help="Disable inverse-frequency class weighting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        select_by=args.select_by,
        class_weighted=args.class_weighted,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
