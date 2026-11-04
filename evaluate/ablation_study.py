"""Ablation study for CHEM.

An ablation study isolates *which* design decisions actually drive CHEM's
performance by removing one component at a time and measuring the resulting drop
in F1 relative to the full model.  A large drop means the component is load
bearing; a negligible drop means it is not.

Four ablations are implemented, matching the interface contract:

1. **no_graph** — remove graph structure entirely.  Node embeddings are mean /
   flat-concatenated and fed to a plain MLP (:class:`FlatMLPVariant`).  Tests the
   central hypothesis that *structure* (not just the bag of elements) matters.
2. **no_policy_tags** — remove the policy-category tags attached to library
   patterns, so the harm-pattern library contributes untyped structural matches
   only.  Tests the value of policy grounding.
3. **no_temporal_edges** — drop ``temporal_sequence`` edges from every graph.
   Tests whether temporal composition contributes to detection.
4. **gcn_instead_of_gat** — replace the attention-based GAT with a standard GCN
   (:class:`GCNVariant` built on ``torch_geometric.nn.GCNConv``).  Tests whether
   attention specifically (vs. plain message passing) is responsible for gains.

Heavy dependencies (torch, torch_geometric, the model/graph packages) are
imported lazily so the module remains importable and AST-parseable in a bare
environment.  ``nn.Module`` subclasses are defined behind a lazy factory so the
file does not require torch at import time.

Public API
----------
run_ablations(...) -> dict
    Run all four ablations and report F1 and F1-drop vs the full model.
build_flat_mlp_variant(...) / build_gcn_variant(...)
    Factories returning the ``FlatMLPVariant`` / ``GCNVariant`` nn.Modules.
main()
    argparse entry point.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from evaluate.metrics import compute_metrics
from utils.embeddings import target_dim


# ---------------------------------------------------------------------------
# Lazy nn.Module variants (defined only when torch is importable)
# ---------------------------------------------------------------------------
def _load_torch_modules():
    """Import torch / torch_geometric building blocks lazily.

    Returns
    -------
    tuple or None
        ``(torch, nn, F, GCNConv, global_mean_pool)`` when the stack is
        available, otherwise ``None`` so callers can degrade gracefully.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool

        return torch, nn, F, GCNConv, global_mean_pool
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[CHEM][ablation] torch/torch_geometric unavailable ({exc})")
        return None


def build_flat_mlp_variant(
    in_dim: Optional[int] = None,
    hidden_dim: int = 256,
    num_classes: int = 2,
) -> Any:
    """Construct the graph-free MLP baseline used by the ``no_graph`` ablation.

    The variant ignores ``edge_index`` entirely: it mean-pools the node feature
    matrix into a single fixed-length content vector and classifies it with a
    two-layer MLP.  This deliberately discards all compositional structure so the
    F1 drop measures how much the graph itself contributes.

    Parameters
    ----------
    in_dim : int, optional
        Node feature dimension.  Defaults to the configured embedding dim (768).
    hidden_dim : int, optional
        Hidden layer width (default 256).
    num_classes : int, optional
        Output classes (default 2: safe / harmful).

    Returns
    -------
    torch.nn.Module
        A ``FlatMLPVariant`` instance.

    Raises
    ------
    RuntimeError
        If torch is not installed.
    """
    modules = _load_torch_modules()
    if modules is None:
        raise RuntimeError("torch is required to build FlatMLPVariant")
    torch, nn, F, _GCNConv, global_mean_pool = modules
    feature_dim = in_dim if in_dim is not None else target_dim()

    class FlatMLPVariant(nn.Module):
        """Graph-free MLP over mean-pooled node embeddings.

        Inputs to ``forward`` mirror the GAT signature ``(x, edge_index, batch)``
        so the same evaluation loop can drive either model, but ``edge_index`` is
        intentionally unused — this is the "remove graph structure" ablation.
        Output is ``log_softmax`` over ``num_classes`` classes, matching the
        full model so metrics are computed identically.
        """

        def __init__(self) -> None:
            """Initialise the two-layer classifier head."""
            super().__init__()
            self.fc1 = nn.Linear(feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x, edge_index, batch):  # noqa: D401 - see class doc
            """Mean-pool node features per graph and classify.

            Parameters
            ----------
            x : Tensor ``[N, feature_dim]``
                Node features for all graphs in the batch.
            edge_index : Tensor
                Ignored (present only for signature compatibility).
            batch : Tensor ``[N]``
                Graph assignment vector used to mean-pool per graph.

            Returns
            -------
            Tensor ``[num_graphs, num_classes]``
                Log-probabilities per graph.
            """
            pooled = global_mean_pool(x, batch)
            hidden = F.relu(self.fc1(pooled))
            hidden = self.dropout(hidden)
            logits = self.fc2(hidden)
            return F.log_softmax(logits, dim=-1)

    return FlatMLPVariant()


def build_gcn_variant(
    in_dim: Optional[int] = None,
    hidden_dim: int = 256,
    num_classes: int = 2,
) -> Any:
    """Construct the GCN baseline used by the ``gcn_instead_of_gat`` ablation.

    Structurally mirrors :class:`CHEMGraphAttentionNetwork` (two graph-conv
    layers, ELU, dropout, global mean pool, linear head, log-softmax) but swaps
    attention-based ``GATConv`` for standard ``GCNConv``.  The F1 drop therefore
    isolates the contribution of attention specifically.

    Parameters
    ----------
    in_dim : int, optional
        Node feature dimension (default: configured embedding dim, 768).
    hidden_dim : int, optional
        Hidden width of the first GCN layer (default 256).
    num_classes : int, optional
        Output classes (default 2).

    Returns
    -------
    torch.nn.Module
        A ``GCNVariant`` instance.

    Raises
    ------
    RuntimeError
        If torch_geometric is not installed.
    """
    modules = _load_torch_modules()
    if modules is None:
        raise RuntimeError("torch_geometric is required to build GCNVariant")
    torch, nn, F, GCNConv, global_mean_pool = modules
    feature_dim = in_dim if in_dim is not None else target_dim()

    class GCNVariant(nn.Module):
        """Two-layer GCN drop-in replacement for the CHEM GAT.

        Uses plain graph-convolution message passing instead of multi-head
        attention.  ``forward(x, edge_index, batch)`` returns ``log_softmax``
        over ``num_classes`` classes so it is interchangeable with the full model
        in the evaluation loop.
        """

        def __init__(self) -> None:
            """Initialise two GCN layers, a pooling head and a classifier."""
            super().__init__()
            self.conv1 = GCNConv(feature_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = nn.Linear(hidden_dim, num_classes)
            self.dropout_p = 0.6

        def forward(self, x, edge_index, batch):  # noqa: D401 - see class doc
            """Run two GCN layers, mean-pool and classify.

            Parameters
            ----------
            x : Tensor ``[N, feature_dim]``
                Node features.
            edge_index : Tensor ``[2, E]``
                Graph connectivity (used, unlike the flat MLP variant).
            batch : Tensor ``[N]``
                Graph assignment vector for pooling.

            Returns
            -------
            Tensor ``[num_graphs, num_classes]``
                Log-probabilities per graph.
            """
            hidden = F.elu(self.conv1(x, edge_index))
            hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)
            hidden = F.elu(self.conv2(hidden, edge_index))
            pooled = global_mean_pool(hidden, batch)
            logits = self.lin(pooled)
            return F.log_softmax(logits, dim=-1)

    return GCNVariant()


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------
def _evaluate_predictions(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Score a set of predictions with the shared metric implementation.

    Parameters
    ----------
    y_true : sequence of int
        Ground-truth binary labels.
    y_prob : sequence of float
        Predicted harmful-class probabilities.
    threshold : float, optional
        Decision threshold (defaults to config ``final_harm_threshold``).

    Returns
    -------
    dict
        The metrics dict from :func:`compute_metrics`.
    """
    return compute_metrics(y_true, y_prob, threshold=threshold)


def _train_and_eval_variant(
    name: str,
    variant_factory: Optional[Callable[[], Any]],
    data_transform: Optional[Callable[[Any], Any]],
    trainer: Optional[Callable[..., Dict[str, Any]]],
    threshold: Optional[float],
) -> Dict[str, float]:
    """Train and evaluate a single ablation variant, returning its metrics.

    The actual training loop is delegated to an injected ``trainer`` callable
    (typically a thin wrapper around ``model.train``) so this module stays free
    of dataset-specific logic.  The ``trainer`` is expected to accept
    ``model=`` and ``data_transform=`` keyword arguments and return a metrics
    dict containing at least ``{"y_true", "y_prob"}`` or ``{"f1"}``.

    Parameters
    ----------
    name : str
        Human-readable ablation name (for logging).
    variant_factory : callable, optional
        Zero-arg factory returning the ablated ``nn.Module`` to train.  ``None``
        means "use the full model architecture" (e.g. tag/edge ablations that
        only change the data, not the model).
    data_transform : callable, optional
        A graph->graph transform applied to every training/eval sample (used by
        the ``no_temporal_edges`` and ``no_policy_tags`` ablations).
    trainer : callable, optional
        Injected training/eval entry point.  When ``None`` the function reports
        that the heavy pipeline is unavailable and returns NaN metrics.
    threshold : float, optional
        Decision threshold forwarded to metric computation.

    Returns
    -------
    dict
        ``{"f1": float, ...}``; ``f1`` is ``nan`` when the variant could not be
        trained/evaluated in the current environment.
    """
    print(f"[CHEM][ablation] --- Running ablation: {name} ---")

    if trainer is None:
        print(
            f"[CHEM][ablation] No trainer supplied for '{name}'. "
            "Inject trainer=... (wrapping model.train) to actually run it. "
            "Reporting NaN so the results table stays well-formed."
        )
        return {"f1": float("nan")}

    try:
        model = variant_factory() if variant_factory is not None else None
        output = trainer(model=model, data_transform=data_transform)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[CHEM][ablation] Ablation '{name}' failed to run ({exc}).")
        return {"f1": float("nan")}

    # Accept either raw predictions or an already-computed f1 from the trainer.
    if isinstance(output, dict) and "y_true" in output and "y_prob" in output:
        metrics = _evaluate_predictions(output["y_true"], output["y_prob"], threshold)
        return metrics
    if isinstance(output, dict) and "f1" in output:
        return {"f1": float(output["f1"])}

    print(f"[CHEM][ablation] Trainer for '{name}' returned an unexpected result.")
    return {"f1": float("nan")}


def _make_no_temporal_edges_transform() -> Callable[[Any], Any]:
    """Return a graph transform that strips ``temporal_sequence`` edges.

    Returns
    -------
    callable
        ``transform(data) -> data`` removing every edge whose ``edge_type``
        equals the ``temporal_sequence`` id (2) from a PyG ``Data`` object.
    """
    def transform(data: Any) -> Any:
        """Remove temporal edges from a single PyG ``Data`` graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with ``edge_index`` and ``edge_type`` attributes.

        Returns
        -------
        torch_geometric.data.Data
            The same graph with temporal edges filtered out.
        """
        try:
            from graph.edge_types import edge_type_id  # noqa: WPS433

            temporal_id = edge_type_id("temporal_sequence")
        except Exception:  # pragma: no cover - defensive
            temporal_id = 2

        if getattr(data, "edge_type", None) is None:
            return data
        keep = data.edge_type != temporal_id
        data.edge_index = data.edge_index[:, keep]
        data.edge_type = data.edge_type[keep]
        return data

    return transform


def _make_no_policy_tags_transform() -> Callable[[Any], Any]:
    """Return a transform that removes policy-category tags from library matches.

    The harm-pattern library normally tags each mined pattern with a policy
    category / clause.  This transform blanks those tags so the model's
    pattern-similarity signal becomes structure-only, isolating the value of
    policy grounding.

    Returns
    -------
    callable
        ``transform(sample) -> sample`` clearing ``policy_category`` /
        ``policy_clause`` fields where present.
    """
    def transform(sample: Any) -> Any:
        """Strip policy tags from a sample or graph in place.

        Parameters
        ----------
        sample : Any
            A PyG ``Data`` object or a dict-like sample carrying policy tags.

        Returns
        -------
        Any
            The same object with policy tags cleared.
        """
        for attr in ("policy_category", "policy_clause"):
            if isinstance(sample, dict):
                sample.pop(attr, None)
            elif hasattr(sample, attr):
                try:
                    setattr(sample, attr, None)
                except Exception:  # pragma: no cover - defensive
                    pass
        # Node-level metadata carried on the graph is also cleared when present.
        node_meta = getattr(sample, "node_meta", None)
        if isinstance(node_meta, list):
            for node in node_meta:
                if isinstance(node, dict):
                    node.pop("policy_category", None)
                    node.pop("policy_clause", None)
        return sample

    return transform


def run_ablations(
    full_model_f1: Optional[float] = None,
    trainer: Optional[Callable[..., Dict[str, Any]]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Run all four CHEM ablations and report F1 and F1-drop vs the full model.

    Parameters
    ----------
    full_model_f1 : float, optional
        The full (un-ablated) model's test F1, used as the reference for the
        drop calculation.  If ``None`` the function first trains/evaluates the
        full model via ``trainer`` (when available) to obtain it.
    trainer : callable, optional
        Injected training/eval entry point wrapping ``model.train`` (see
        :func:`_train_and_eval_variant`).  When ``None`` the study still runs
        end to end, reporting NaN metrics and documenting the flow — useful for
        wiring and CI where the heavy stack is absent.
    threshold : float, optional
        Decision threshold forwarded to metric computation.

    Returns
    -------
    dict
        ``{ablation_name: {"f1": float, "f1_drop": float}}`` for each of the
        four ablations.  ``f1_drop = full_model_f1 - ablation_f1`` (positive =
        the component helped); ``nan`` where a value is unavailable.
    """
    print("=" * 70)
    print("[CHEM][ablation] CHEM ablation study — four components")
    print("=" * 70)

    # --- Reference: full model F1 --------------------------------------------
    if full_model_f1 is None and trainer is not None:
        print("[CHEM][ablation] Establishing full-model reference F1...")
        try:
            full_output = trainer(model=None, data_transform=None)
            if isinstance(full_output, dict) and "y_true" in full_output:
                full_model_f1 = _evaluate_predictions(
                    full_output["y_true"], full_output["y_prob"], threshold
                )["f1"]
            elif isinstance(full_output, dict) and "f1" in full_output:
                full_model_f1 = float(full_output["f1"])
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[CHEM][ablation] Could not establish full-model F1 ({exc}).")

    if full_model_f1 is None:
        print(
            "[CHEM][ablation] No full-model F1 available; f1_drop will be NaN. "
            "Pass full_model_f1= or a trainer= to compute it."
        )

    # --- Define the four ablations -------------------------------------------
    # Each entry: (name, variant_factory, data_transform).  A None factory means
    # the model architecture is unchanged (only the data is ablated).
    ablation_specs: List[Any] = [
        ("no_graph", build_flat_mlp_variant, None),
        ("no_policy_tags", None, _make_no_policy_tags_transform()),
        ("no_temporal_edges", None, _make_no_temporal_edges_transform()),
        ("gcn_instead_of_gat", build_gcn_variant, None),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, factory, transform in ablation_specs:
        metrics = _train_and_eval_variant(
            name=name,
            variant_factory=factory,
            data_transform=transform,
            trainer=trainer,
            threshold=threshold,
        )
        f1 = float(metrics.get("f1", float("nan")))
        if full_model_f1 is not None and not np.isnan(f1):
            drop = float(full_model_f1) - f1
        else:
            drop = float("nan")
        results[name] = {"f1": f1, "f1_drop": drop}

    _print_results_table(results, full_model_f1)
    return results


def _print_results_table(
    results: Dict[str, Dict[str, float]],
    full_model_f1: Optional[float],
) -> None:
    """Print an aligned results table for the ablation study.

    Parameters
    ----------
    results : dict
        The ``{ablation_name: {"f1", "f1_drop"}}`` mapping from
        :func:`run_ablations`.
    full_model_f1 : float, optional
        The reference full-model F1 (printed in the header for context).
    """
    ref = "n/a" if full_model_f1 is None else f"{full_model_f1:.4f}"
    print("\n" + "=" * 60)
    print(f"[CHEM][ablation] RESULTS  (full-model F1 = {ref})")
    print("=" * 60)
    header = f"{'ablation':<24}{'f1':>10}{'f1_drop':>12}"
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        f1 = row.get("f1", float("nan"))
        drop = row.get("f1_drop", float("nan"))
        print(f"{name:<24}{f1:>10.4f}{drop:>12.4f}")
    print("=" * 60)


def _make_trainer(data_dir: str, epochs: int, batch_size: int) -> Callable:
    """Build a trainer closure that integrates with the model.train pipeline.

    The returned callable accepts ``model=`` (an ``nn.Module`` or ``None`` to use
    the default GAT) and ``data_transform=`` (an optional graph->graph transform)
    and returns ``{"f1": float}`` after training and evaluating on the dataset.

    Parameters
    ----------
    data_dir : str
        Hateful Memes dataset root directory.
    epochs : int
        Maximum training epochs per variant.
    batch_size : int
        Graphs per mini-batch.

    Returns
    -------
    callable
        A ``trainer(model, data_transform) -> dict`` closure.
    """
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    from model.train import _load_samples, _build_dataset, _split_dataset, _compute_metrics
    from encoders.text_encoder import TextEncoder
    from encoders.image_encoder import ImageEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoders and dataset once; all variants share the same graphs.
    print("[CHEM][ablation] Loading encoders (shared across variants)...")
    text_enc = TextEncoder(device=str(device))
    image_enc = ImageEncoder(device=str(device))

    print("[CHEM][ablation] Loading and encoding dataset (done once)...")
    samples = _load_samples("train", data_dir) + _load_samples("dev", data_dir)
    full_dataset = _build_dataset(samples, text_enc, image_enc)
    splits = _split_dataset(full_dataset)
    print(f"[CHEM][ablation] Dataset ready: {len(full_dataset)} graphs.")

    def trainer(model=None, data_transform=None):
        """Train a variant model and return its test F1.

        Parameters
        ----------
        model : nn.Module or None
            Model to train. ``None`` uses the default full GAT.
        data_transform : callable or None
            Optional per-graph transform applied to every sample before batching.

        Returns
        -------
        dict
            ``{"f1": float}`` from test-set evaluation.
        """
        from model.gat_model import CHEMGraphAttentionNetwork

        if model is None:
            model = CHEMGraphAttentionNetwork()
        model = model.to(device)

        # Apply data transform if supplied (e.g. strip temporal edges).
        def _apply(dataset):
            if data_transform is None:
                return dataset
            return [data_transform(g) for g in dataset]

        train_loader = DataLoader(_apply(splits["train"]), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(_apply(splits["val"]),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(_apply(splits["test"]),  batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.NLLLoss()
        best_val_f1, no_improve = -1.0, 0

        for epoch in range(1, epochs + 1):
            # --- train ---
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x.float(), batch.edge_index, batch.batch)
                criterion(out, batch.y.view(-1).long()).backward()
                optimizer.step()
            # --- val ---
            model.eval()
            y_true, y_prob = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    probs = torch.exp(model(batch.x.float(), batch.edge_index, batch.batch))[:, 1]
                    y_prob.extend(probs.cpu().tolist())
                    y_true.extend(batch.y.cpu().tolist())
            val_f1 = _compute_metrics(y_true, y_prob).get("f1", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 20:
                    print(f"[CHEM][ablation]   early stop at epoch {epoch}")
                    break
            if epoch % 5 == 0 or epoch == 1:
                print(f"[CHEM][ablation]   epoch {epoch}/{epochs}  val_f1={val_f1:.4f}")

        # --- test ---
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                probs = torch.exp(model(batch.x.float(), batch.edge_index, batch.batch))[:, 1]
                y_prob.extend(probs.cpu().tolist())
                y_true.extend(batch.y.cpu().tolist())
        metrics = _compute_metrics(y_true, y_prob)
        print(f"[CHEM][ablation]   test_f1={metrics.get('f1', 0.0):.4f}")
        return {"f1": metrics.get("f1", 0.0)}

    return trainer


def main() -> None:
    """Command-line entry point for the ablation study.

    Wires the real training pipeline into all four ablation variants and prints a
    results table showing F1 and F1-drop vs the full model.  Pass
    ``--full-model-f1`` to use an already-known baseline and skip retraining it.
    """
    parser = argparse.ArgumentParser(
        description="CHEM ablation study (no-graph / no-tags / no-temporal / GCN).",
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
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs per ablation variant (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Graphs per mini-batch (default: 32).",
    )
    parser.add_argument(
        "--full-model-f1",
        type=float,
        default=None,
        help="Known full-model test F1 (skip retraining the baseline). "
             "E.g. pass 0.5544 if you already trained the full model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: config final_harm_threshold).",
    )
    args = parser.parse_args()

    trainer = _make_trainer(args.data_dir, args.epochs, args.batch_size)
    run_ablations(
        full_model_f1=args.full_model_f1,
        trainer=trainer,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
