"""Small, testable helpers for ``model.train``.

* class weights for an imbalanced binary label set,
* decision-threshold selection on a validation set,
* metrics at an arbitrary threshold,
* an on-disk cache of built graphs, keyed by a fingerprint of the samples *and*
  of the source files that shape a graph, so editing an encoder or the graph
  builder automatically invalidates stale graphs.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Files whose contents change what a built graph looks like.
_GRAPH_SHAPING_FILES = (
    "encoders/text_encoder.py",
    "encoders/image_encoder.py",
    "graph/graph_builder.py",
    "policy/policy_config.json",
)


# ------------------------------------------------------------------ weights
def class_weights(labels: Iterable[int]) -> List[float]:
    """Inverse-frequency weights ``n / (2 * n_c)`` for classes 0 and 1.

    A class with no examples gets weight 1.0 so the loss stays defined.
    """
    labels = [int(l) for l in labels]
    n = len(labels)
    counts = [labels.count(0), labels.count(1)]
    return [n / (2.0 * c) if c else 1.0 for c in counts]


# ---------------------------------------------------------------- threshold
def metrics_at_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """accuracy / precision / recall / F1 / AUC with ``p >= threshold`` => 1."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


def best_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], metric: str = "f1"
) -> float:
    """Grid-search the threshold (0.05..0.95, step 0.01) maximising ``metric``.

    Meant to be called on VALIDATION predictions only; ties keep the threshold
    closest to 0.5 so the result is stable.
    """
    if metric not in ("f1", "accuracy"):
        raise ValueError("metric must be 'f1' or 'accuracy'")
    best_t, best_v = 0.5, -1.0
    for i in range(5, 96):
        t = i / 100.0
        v = metrics_at_threshold(y_true, y_prob, t)[metric]
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = t, v
    return best_t


# -------------------------------------------------------------------- cache
def graph_cache_key(samples: Sequence[Dict[str, Any]], root: str = _ROOT) -> str:
    """Fingerprint of the samples plus the files that shape a graph."""
    h = hashlib.sha256()
    for s in samples:
        h.update(repr((s.get("id"), s.get("img"), s.get("text"), s.get("label"))).encode("utf-8"))
    for rel in _GRAPH_SHAPING_FILES:
        path = os.path.join(root, rel)
        h.update(rel.encode())
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"graphs_{key}.pt")


def load_graph_cache(cache_dir: Optional[str], key: str) -> Optional[List[Any]]:
    """Return cached graphs for ``key`` or ``None`` (missing / unreadable)."""
    if not cache_dir:
        return None
    path = cache_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    import torch

    try:
        # Files in this folder are written by save_graph_cache() on this machine;
        # PyG Data objects need full unpickling, so never point this at a file
        # you did not create.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[CHEM][train] cache unreadable ({exc}); rebuilding.")
        return None


def save_graph_cache(cache_dir: Optional[str], key: str, graphs: List[Any]) -> Optional[str]:
    if not cache_dir:
        return None
    import torch

    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, key)
    tmp = path + ".tmp"
    torch.save(graphs, tmp)
    os.replace(tmp, path)  # never leave a half-written cache behind
    return path


def subsample(samples: List[Dict[str, Any]], max_samples: Optional[int], seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministic random subset (all samples when ``max_samples`` is falsy)."""
    if not max_samples or max_samples >= len(samples):
        return samples
    import random

    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    return [samples[i] for i in sorted(idx[:max_samples])]
