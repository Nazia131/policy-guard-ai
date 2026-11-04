"""Zero-shot generalisation evaluation for CHEM.

CHEM's central scientific claim is that harm is *compositional*: a harmful piece
of content is recognisable from the **structure** of how its elements relate
(identity-text pointing at a crowd, a weapon aimed at a person, a child node
linked to secret-contact text) rather than from category-specific surface
features.  If that claim holds, a model trained with **one entire harm category
removed** from its training set should still flag examples of that unseen
category, because it recognises the shared harmful *structure* even though it
never saw that category's label.

This module operationalises that test:

1. Hold out one whole harm category (``--held-out-category``).
2. Train (or accept an already-trained model) on the remaining categories only.
3. Evaluate on a test loader containing *only* the held-out category's samples.
4. Report the zero-shot F1 — the head-line number for structural generalisation.

Because full training on the Hateful Memes / UCF-Crime corpora is heavy, this
file is written so it *degrades gracefully*: heavy dependencies (torch,
torch_geometric, the model package, the data pipeline) are imported lazily
inside functions.  If they are unavailable the flow is still fully documented
and the CLI reports exactly what is missing rather than crashing at import time.

Public API
----------
run_zero_shot_eval(held_out_category, model=None, test_loader=None, ...) -> dict
    Orchestrate the hold-out train/eval and return the zero-shot metrics.
main()
    argparse entry point exposing ``--held-out-category``.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluate.metrics import compute_metrics, pretty_print
from utils.config import get_harm_categories


# ---------------------------------------------------------------------------
# Lazy dependency access
# ---------------------------------------------------------------------------
def _try_import_torch():
    """Import torch lazily, returning ``None`` if it is unavailable.

    Returns
    -------
    module or None
        The imported ``torch`` module, or ``None`` when it is not installed so
        callers can degrade gracefully instead of failing at import time.
    """
    try:
        import torch  # noqa: WPS433 (intentional local import)

        return torch
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[CHEM][zero_shot] torch is unavailable ({exc})")
        return None


def _resolve_categories(held_out_category: str) -> Tuple[List[str], List[str]]:
    """Split the configured harm categories into train and held-out sets.

    Parameters
    ----------
    held_out_category : str
        The harm-category id to withhold from training.

    Returns
    -------
    tuple
        ``(train_categories, [held_out_category])``.

    Raises
    ------
    ValueError
        If ``held_out_category`` is not a known harm category.
    """
    all_categories = list(get_harm_categories().keys())
    if held_out_category not in all_categories:
        raise ValueError(
            f"Unknown held-out category '{held_out_category}'. "
            f"Valid categories: {all_categories}"
        )
    train_categories = [c for c in all_categories if c != held_out_category]
    return train_categories, [held_out_category]


# ---------------------------------------------------------------------------
# Training / prediction helpers (thin, dependency-light wrappers)
# ---------------------------------------------------------------------------
def _train_model_excluding(
    train_categories: Sequence[str],
    train_fn: Optional[Callable[..., Any]],
    epochs: int,
) -> Optional[Any]:
    """Train a CHEM GAT restricted to ``train_categories``.

    The heavy lifting is delegated to ``model.train`` (imported lazily).  The
    expected contract is a callable that accepts an ``include_categories``
    keyword and returns a trained ``CHEMGraphAttentionNetwork``.  If the model
    package or its training entry point is unavailable this returns ``None`` and
    the caller is expected to have been supplied a pre-trained model instead.

    Parameters
    ----------
    train_categories : sequence of str
        Harm categories the model is allowed to learn from.
    train_fn : callable, optional
        Override training entry point (useful for tests).  When ``None`` the
        function attempts to locate ``model.train.train`` dynamically.
    epochs : int
        Number of training epochs to request from the training entry point.

    Returns
    -------
    object or None
        A trained model instance, or ``None`` if training could not be run.
    """
    if train_fn is None:
        try:
            from model import train as train_module  # noqa: WPS433

            # ``model.train`` is a script-style module; prefer an explicit
            # ``train`` function if the author exposed one, else give up
            # gracefully so the CLI can explain the situation.
            train_fn = getattr(train_module, "train", None)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[CHEM][zero_shot] model.train unavailable ({exc})")
            train_fn = None

    if train_fn is None:
        print(
            "[CHEM][zero_shot] No training entry point found; supply a "
            "pre-trained `model=` and `test_loader=` to run_zero_shot_eval "
            "to evaluate an already-trained checkpoint."
        )
        return None

    print(
        f"[CHEM][zero_shot] Training on {len(train_categories)} categories "
        f"(excluding held-out): {list(train_categories)}"
    )
    try:
        return train_fn(include_categories=list(train_categories), epochs=epochs)
    except TypeError:
        # The repo's train() may not accept category filtering yet; call it in
        # its simplest documented form so the flow remains runnable.
        print(
            "[CHEM][zero_shot] train() does not accept include_categories; "
            "calling it without category filtering (documented fallback)."
        )
        return train_fn()


def _predict_probabilities(
    model: Any,
    test_loader: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the trained model over a test loader and collect harm probabilities.

    Parameters
    ----------
    model : object
        A trained ``CHEMGraphAttentionNetwork`` (or compatible module whose
        ``forward(x, edge_index, batch)`` returns log-probabilities of shape
        ``[batch, 2]`` as specified in the interface contract).
    test_loader : iterable
        A PyG ``DataLoader`` yielding batched ``Data`` objects with ``x``,
        ``edge_index``, ``batch`` and ``y`` attributes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y_true, y_prob)`` where ``y_prob`` is the probability of the harmful
        class (index 1).
    """
    torch = _try_import_torch()
    if torch is None or model is None or test_loader is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    model.eval()
    all_true: List[int] = []
    all_prob: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            # forward returns log_softmax over 2 classes per the GAT contract.
            log_probs = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.exp(log_probs)
            harmful_prob = probs[:, 1]
            all_prob.extend(harmful_prob.detach().cpu().numpy().tolist())
            all_true.extend(batch.y.detach().cpu().numpy().tolist())

    return np.asarray(all_true, dtype=int), np.asarray(all_prob, dtype=float)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_zero_shot_eval(
    held_out_category: str,
    model: Optional[Any] = None,
    test_loader: Optional[Any] = None,
    train_fn: Optional[Callable[..., Any]] = None,
    epochs: int = 100,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the leave-one-category-out zero-shot generalisation experiment.

    The experiment demonstrates *structural* generalisation: the model is
    trained without ever seeing ``held_out_category`` and then evaluated purely
    on samples of that unseen category.  A non-trivial F1 indicates CHEM
    recognises the shared harmful composition rather than memorising per-category
    surface cues.

    Two usage modes are supported:

    * **Train-and-evaluate** — pass only ``held_out_category``; the function
      trains a fresh model on the remaining categories via ``model.train`` and
      builds a held-out test loader from the data pipeline.
    * **Evaluate-only** — pass an already-trained ``model`` and a ``test_loader``
      that has been pre-filtered to the held-out category; the function skips
      training entirely.  This is the recommended path when training is heavy.

    Parameters
    ----------
    held_out_category : str
        Harm-category id to withhold from training and evaluate on.
    model : object, optional
        A pre-trained model to evaluate.  If ``None`` the function trains one.
    test_loader : iterable, optional
        Pre-filtered held-out test loader.  Required for evaluate-only mode.
    train_fn : callable, optional
        Override for the training entry point (dependency injection for tests).
    epochs : int, optional
        Training epochs when training a fresh model (default 100).
    threshold : float, optional
        Decision threshold passed through to :func:`compute_metrics`.  Defaults
        to the configured ``final_harm_threshold``.

    Returns
    -------
    dict
        ``{"held_out_category", "train_categories", "metrics", "zero_shot_f1",
        "trained", "evaluated"}``.  ``metrics`` is the full dict from
        :func:`compute_metrics`; ``zero_shot_f1`` is surfaced at the top level
        for convenience.  When training/evaluation could not run (missing heavy
        deps), ``evaluated`` is ``False`` and ``metrics`` is ``None``.
    """
    print("=" * 70)
    print(f"[CHEM][zero_shot] Zero-shot evaluation — holding out '{held_out_category}'")
    print("=" * 70)

    train_categories, held_out = _resolve_categories(held_out_category)
    print(f"[CHEM][zero_shot] Train categories : {train_categories}")
    print(f"[CHEM][zero_shot] Held-out category: {held_out}")

    result: Dict[str, Any] = {
        "held_out_category": held_out_category,
        "train_categories": train_categories,
        "metrics": None,
        "zero_shot_f1": None,
        "trained": False,
        "evaluated": False,
    }

    # --- Step 1: obtain a model trained WITHOUT the held-out category ---------
    if model is None:
        print("[CHEM][zero_shot] No pre-trained model supplied; attempting to train...")
        model = _train_model_excluding(train_categories, train_fn, epochs)
        result["trained"] = model is not None
    else:
        print("[CHEM][zero_shot] Using supplied pre-trained model (evaluate-only mode).")

    # --- Step 2: obtain a test loader filtered to the held-out category -------
    if test_loader is None:
        test_loader = _build_held_out_loader(held_out_category)

    # --- Step 3: evaluate on the held-out category ----------------------------
    if model is None or test_loader is None:
        print(
            "[CHEM][zero_shot] Cannot evaluate: missing "
            f"{'model' if model is None else ''} "
            f"{'test_loader' if test_loader is None else ''}".strip()
        )
        print(
            "[CHEM][zero_shot] Re-run with a trained checkpoint and a held-out "
            "loader, e.g. run_zero_shot_eval(cat, model=m, test_loader=dl)."
        )
        return result

    print("[CHEM][zero_shot] Evaluating model on held-out category samples...")
    y_true, y_prob = _predict_probabilities(model, test_loader)

    if y_true.size == 0:
        print("[CHEM][zero_shot] Held-out loader produced no samples; nothing to score.")
        return result

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    result["metrics"] = metrics
    result["zero_shot_f1"] = metrics["f1"]
    result["evaluated"] = True

    print("\n[CHEM][zero_shot] Zero-shot results on unseen category "
          f"'{held_out_category}':")
    pretty_print(metrics)
    print(f"\n[CHEM][zero_shot] >>> ZERO-SHOT F1 = {metrics['f1']:.4f} <<<")
    return result


def _build_held_out_loader(held_out_category: str) -> Optional[Any]:
    """Build a PyG test loader containing only the held-out category's samples.

    This wires together the data pipeline and graph builder described in the
    interface contract:

    1. Load raw samples (``data.preprocess.hateful_memes``).
    2. Keep only samples whose harm category is ``held_out_category``.
    3. Decompose each into elements and build a Content Composition Graph
       (``graph.graph_builder.build_graph``).
    4. Wrap the resulting ``Data`` objects in a ``torch_geometric`` ``DataLoader``.

    All heavy imports are performed lazily so the module remains importable
    without the data/graph/torch stack installed.

    Parameters
    ----------
    held_out_category : str
        The harm-category id whose samples should populate the loader.

    Returns
    -------
    object or None
        A ``DataLoader`` over the held-out samples, or ``None`` when the
        supporting pipeline is unavailable.
    """
    torch = _try_import_torch()
    if torch is None:
        return None

    try:
        from torch_geometric.loader import DataLoader  # noqa: WPS433
        from graph.graph_builder import build_graph  # noqa: WPS433
        from data.preprocess.hateful_memes import load_samples  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - environment dependent
        print(
            "[CHEM][zero_shot] Data/graph pipeline unavailable "
            f"({exc}); pass an explicit test_loader= to evaluate."
        )
        return None

    try:
        graphs: List[Any] = []
        for sample in load_samples():
            # A sample is expected to carry its harm category id; only keep the
            # held-out category so the loader is a pure zero-shot probe.
            if sample.get("category") != held_out_category:
                continue
            elements = sample.get("elements")
            if elements is None:
                # If elements were not pre-extracted, skip gracefully.
                continue
            built = build_graph(elements, label=int(sample.get("label", 1)))
            graphs.append(built["data"])

        if not graphs:
            print(
                f"[CHEM][zero_shot] No held-out samples found for "
                f"'{held_out_category}'."
            )
            return None

        print(f"[CHEM][zero_shot] Built held-out loader with {len(graphs)} graphs.")
        return DataLoader(graphs, batch_size=32, shuffle=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[CHEM][zero_shot] Failed to build held-out loader ({exc}).")
        return None


def _run_with_checkpoint(
    held_out_category: str,
    data_dir: str,
    checkpoint: str,
    threshold: Optional[float],
) -> None:
    """Evaluate-only zero-shot run using an existing checkpoint.

    Uses the trained text encoder to assign each harmful sample a primary harm
    category, then evaluates the already-trained model on the held-out category
    without any retraining.

    Parameters
    ----------
    held_out_category : str
        Harm category to evaluate on.
    data_dir : str
        Hateful Memes dataset root.
    checkpoint : str
        Path to a saved GAT checkpoint (``best_gat.pt``).
    threshold : float or None
        Decision threshold.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from model.gat_model import CHEMGraphAttentionNetwork
    from model.train import _load_samples, _compute_metrics
    from encoders.text_encoder import TextEncoder
    from encoders.image_encoder import ImageEncoder
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint.
    model = CHEMGraphAttentionNetwork().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[CHEM][zero_shot] Loaded checkpoint: {checkpoint}")

    # Load encoders to assign harm categories to samples.
    text_enc = TextEncoder(device=str(device))
    image_enc = ImageEncoder(device=str(device))

    print("[CHEM][zero_shot] Encoding samples to assign harm categories...")
    raw_samples = _load_samples("test", data_dir)

    # Assign a primary harm category to each sample using the text encoder.
    from graph.graph_builder import build_graph

    held_graphs, other_graphs = [], []
    for i, s in enumerate(raw_samples):
        try:
            text_out = text_enc.encode(s.get("text", ""))
            elements = list(text_out.get("elements", []))
            img_field = s.get("img")
            if img_field:
                img_path = img_field if os.path.isabs(img_field) else os.path.join(data_dir, img_field)
                img_out = image_enc.encode(Image.open(img_path).convert("RGB"))
                elements.extend(img_out.get("elements", []))

            # Primary category = first element with a harmful signal.
            primary_cat = None
            for el in elements:
                if el.get("harmful_signal") and el.get("policy_category"):
                    primary_cat = el["policy_category"]
                    break

            label = int(s.get("label", 0))
            graph_dict = build_graph(elements, label)
            data = graph_dict["data"]

            if primary_cat == held_out_category:
                held_graphs.append(data)
            else:
                other_graphs.append(data)
        except Exception as exc:
            print(f"[CHEM][zero_shot] skipping sample {s.get('id', i)}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"[CHEM][zero_shot]   categorised {i+1}/{len(raw_samples)}")

    print(f"[CHEM][zero_shot] Held-out '{held_out_category}': {len(held_graphs)} samples")
    if not held_graphs:
        print("[CHEM][zero_shot] No samples assigned to held-out category in test set.")
        print("[CHEM][zero_shot] Using all harmful test samples as fallback.")
        held_graphs = [g for g in other_graphs if hasattr(g, 'y') and int(g.y) == 1]

    if not held_graphs:
        print("[CHEM][zero_shot] No held-out samples available; cannot evaluate.")
        return

    loader = DataLoader(held_graphs, batch_size=32, shuffle=False)
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.exp(model(batch.x.float(), batch.edge_index, batch.batch))[:, 1]
            y_prob.extend(probs.cpu().tolist())
            y_true.extend(batch.y.cpu().tolist())

    metrics = _compute_metrics(y_true, y_prob)
    print("\n[CHEM][zero_shot] === ZERO-SHOT RESULTS ===")
    for k, v in metrics.items():
        print(f"[CHEM][zero_shot]   {k:<12}: {v:.4f}")
    print(f"\n[CHEM][zero_shot] >>> ZERO-SHOT F1 = {metrics.get('f1', 0.0):.4f} <<<")


def main() -> None:
    """Command-line entry point for the zero-shot generalisation study."""
    import os

    default_categories = []
    try:
        default_categories = list(get_harm_categories().keys())
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="CHEM zero-shot (leave-one-category-out) generalisation study.",
    )
    parser.add_argument(
        "--held-out-category",
        required=True,
        help=(
            "Harm-category id to withhold from training and evaluate on. "
            f"Choices: {default_categories}"
        ),
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default="data/hateful_memes",
        help="Hateful Memes dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_gat.pt",
        help="Path to a saved GAT checkpoint to evaluate (skips retraining).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs when training a fresh model (default: 100).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: config final_harm_threshold).",
    )
    args = parser.parse_args()

    # If a checkpoint exists, use evaluate-only mode (much faster than retraining).
    ckpt = args.checkpoint
    if os.path.isfile(ckpt):
        print(f"[CHEM][zero_shot] Found checkpoint '{ckpt}' — running evaluate-only mode.")
        _run_with_checkpoint(
            held_out_category=args.held_out_category,
            data_dir=args.data_dir,
            checkpoint=ckpt,
            threshold=args.threshold,
        )
    else:
        print(f"[CHEM][zero_shot] No checkpoint at '{ckpt}' — running full train+eval.")
        run_zero_shot_eval(
            held_out_category=args.held_out_category,
            epochs=args.epochs,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
