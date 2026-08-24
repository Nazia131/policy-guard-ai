"""PyTorch-Geometric GNNExplainer wrapper for the CHEM GAT.

Where :mod:`explainability.shap_explainer` attributes a decision to *node
features*, this module answers the complementary question: *which subgraph
(nodes AND edges) is minimally responsible for the classification?*  It uses
PyTorch Geometric's :class:`torch_geometric.explain.GNNExplainer`, which learns
soft masks over nodes and edges by maximising the mutual information between the
masked subgraph and the model's original prediction.

API compatibility
------------------
PyG's explainability API changed shape across releases:

* Newer PyG (>= 2.3) exposes a unified
  :class:`torch_geometric.explain.Explainer` façade that is *configured* with an
  algorithm, e.g.
  ``Explainer(model, algorithm=GNNExplainer(epochs=...), explanation_type=...,
  node_mask_type=..., edge_mask_type=..., model_config=...)``.  Calling the
  explainer returns an ``Explanation`` object carrying ``node_mask`` /
  ``edge_mask`` tensors.
* Older PyG (< 2.3) exposed ``torch_geometric.nn.GNNExplainer`` directly with an
  ``explain_graph`` / ``explain_node`` method returning ``(node_feat_mask,
  edge_mask)``.

This wrapper prefers the newer ``torch_geometric.explain`` API and transparently
falls back to the legacy ``torch_geometric.nn.GNNExplainer`` when only that is
available.  If neither is importable (dependency missing) it returns an
informative degraded result rather than raising.

Public API
----------
``GNNExplainerWrapper(model, epochs=200, device=None)``
``GNNExplainerWrapper.explain(graph_dict) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

# --- PyG explainability imports (declared at top; degrade if unavailable) ----
# We attempt the modern unified API first, then the legacy one.  Each attempt is
# recorded so ``explain`` can report exactly which backend (if any) was used.
_NEW_API_AVAILABLE = False
_LEGACY_API_AVAILABLE = False
_PYG_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - depends on installed PyG version
    from torch_geometric.explain import Explainer, GNNExplainer  # type: ignore
    from torch_geometric.explain.config import ModelConfig  # type: ignore

    _NEW_API_AVAILABLE = True
except Exception as _new_exc:  # noqa: F841
    _PYG_IMPORT_ERROR = _new_exc
    try:  # pragma: no cover - legacy fallback
        from torch_geometric.nn import GNNExplainer as LegacyGNNExplainer  # type: ignore

        _LEGACY_API_AVAILABLE = True
    except Exception as _legacy_exc:  # noqa: F841
        _PYG_IMPORT_ERROR = _legacy_exc

from model.gat_model import CHEMGraphAttentionNetwork  # noqa: E402  (absolute import per spec)


class GNNExplainerWrapper:
    """Find the minimal subgraph responsible for a CHEM harm classification.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        A trained GAT.  Put in eval mode; never trained here.
    epochs : int
        Number of optimisation steps GNNExplainer runs to learn the masks.
    device : str or torch.device, optional
        Device to run explanation on.  Defaults to CUDA if available else CPU.

    Notes
    -----
    The wrapper supports both the modern ``torch_geometric.explain.Explainer``
    (algorithm = ``GNNExplainer``) API and the legacy
    ``torch_geometric.nn.GNNExplainer`` API; see the module docstring.
    """

    def __init__(
        self,
        model: CHEMGraphAttentionNetwork,
        epochs: int = 200,
        device: Optional[Any] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.epochs = int(epochs)
        self._backend = (
            "explain_api"
            if _NEW_API_AVAILABLE
            else ("legacy_nn" if _LEGACY_API_AVAILABLE else "unavailable")
        )
        print(
            f"[CHEM][gnn_explainer] Initialised (device={self.device}, "
            f"epochs={self.epochs}, backend={self._backend})"
        )

    # -- internal helpers ---------------------------------------------------

    def _extract(self, graph_dict: Dict[str, Any]):
        """Pull the tensors and node metadata needed for explanation.

        Parameters
        ----------
        graph_dict : dict
            CHEM graph dict with a ``"data"`` PyG object (``x``, ``edge_index``)
            and optional ``"nodes"``/``node_meta`` metadata and ``"edges"``.

        Returns
        -------
        tuple
            ``(x, edge_index, batch, node_meta, edge_list)``.

        Raises
        ------
        KeyError
            If no usable ``data`` object is present.
        """
        data = graph_dict.get("data")
        if data is None:
            raise KeyError("graph_dict must contain a 'data' PyG object with x/edge_index")
        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device).long()
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        node_meta = getattr(data, "node_meta", None) or graph_dict.get("nodes")
        if node_meta is None:
            node_meta = [{"id": f"node_{i}"} for i in range(x.shape[0])]
        edge_list = graph_dict.get("edges", [])
        return x, edge_index, batch, node_meta, edge_list

    def _threshold_indices(self, mask: List[float], quantile: float = 0.7) -> List[int]:
        """Return indices whose mask value exceeds a quantile threshold.

        Parameters
        ----------
        mask : list[float]
            Importance mask values (per node or per edge).
        quantile : float
            Fraction below which entries are considered unimportant.

        Returns
        -------
        list[int]
            Indices of the "important" entries (empty if the mask is empty).
        """
        if not mask:
            return []
        ordered = sorted(mask)
        cut = ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]
        # Keep strictly-positive, above-threshold entries so an all-zero mask
        # yields no spurious "important" elements.
        return [i for i, v in enumerate(mask) if v >= cut and v > 0.0]

    def _degraded(self, note: str, num_nodes: int = 0, num_edges: int = 0) -> Dict[str, Any]:
        """Build an informative empty result when explanation cannot run.

        Parameters
        ----------
        note : str
            Human-readable reason for degradation.
        num_nodes, num_edges : int
            Sizes used to emit zero-filled masks of the right length.

        Returns
        -------
        dict
            The standard result schema with empty masks and ``available=False``.
        """
        return {
            "node_mask": [0.0] * num_nodes,
            "edge_mask": [0.0] * num_edges,
            "important_nodes": [],
            "important_edges": [],
            "backend": self._backend,
            "available": False,
            "note": note,
        }

    # -- public API ---------------------------------------------------------

    def explain(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute node and edge importance masks for the classification.

        Parameters
        ----------
        graph_dict : dict
            A CHEM graph dict from ``ContentCompositionGraphBuilder.build``.

        Returns
        -------
        dict
            ``{
                "node_mask":       list[float],  # per-node importance
                "edge_mask":       list[float],  # per-edge importance
                "important_nodes": list[dict],   # {index,id,label,score}
                "important_edges": list[dict],   # {index,source,target,score}
                "backend":         str,          # explain_api|legacy_nn|unavailable
                "available":       bool,         # whether GNNExplainer ran
                "note":            str,          # status / method description
            }``
            Never raises; on any failure a degraded result is returned.
        """
        try:
            x, edge_index, batch, node_meta, edge_list = self._extract(graph_dict)
        except Exception as exc:
            print(f"[CHEM][gnn_explainer] Could not read graph: {exc}")
            return self._degraded(f"Invalid graph_dict: {exc}")

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if self._backend == "unavailable":
            print(
                "[CHEM][gnn_explainer] torch_geometric.explain not available "
                f"({_PYG_IMPORT_ERROR}); returning degraded result."
            )
            return self._degraded(
                f"PyG explainability unavailable: {_PYG_IMPORT_ERROR}", num_nodes, num_edges
            )

        try:
            if self._backend == "explain_api":
                node_mask, edge_mask = self._run_new_api(x, edge_index, batch)
                method_note = "torch_geometric.explain.Explainer(GNNExplainer)."
            else:
                node_mask, edge_mask = self._run_legacy_api(x, edge_index, batch)
                method_note = "legacy torch_geometric.nn.GNNExplainer."
        except Exception as exc:
            print(f"[CHEM][gnn_explainer] Explanation failed: {exc}")
            return self._degraded(f"GNNExplainer raised '{exc}'", num_nodes, num_edges)

        # Normalise masks to plain python floats aligned with the graph.
        node_mask = [float(v) for v in node_mask][:num_nodes]
        edge_mask = [float(v) for v in edge_mask][:num_edges]
        if len(node_mask) < num_nodes:
            node_mask += [0.0] * (num_nodes - len(node_mask))
        if len(edge_mask) < num_edges:
            edge_mask += [0.0] * (num_edges - len(edge_mask))

        important_node_idx = self._threshold_indices(node_mask)
        important_edge_idx = self._threshold_indices(edge_mask)

        important_nodes = [
            {
                "index": i,
                "id": str(node_meta[i].get("id", f"node_{i}")) if i < len(node_meta) else f"node_{i}",
                "label": str(node_meta[i].get("full_label", node_meta[i].get("label", "")))
                if i < len(node_meta)
                else "",
                "score": node_mask[i],
            }
            for i in important_node_idx
        ]

        ei = edge_index.cpu().tolist()
        important_edges = [
            {
                "index": i,
                "source": int(ei[0][i]),
                "target": int(ei[1][i]),
                "score": edge_mask[i],
            }
            for i in important_edge_idx
        ]

        print(
            f"[CHEM][gnn_explainer] Done: {len(important_nodes)} important nodes, "
            f"{len(important_edges)} important edges (backend={self._backend})."
        )
        return {
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "important_nodes": important_nodes,
            "important_edges": important_edges,
            "backend": self._backend,
            "available": True,
            "note": method_note,
        }

    # -- backend implementations -------------------------------------------

    def _run_new_api(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """Run explanation via the modern ``torch_geometric.explain`` API.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, F]`` node features.
        edge_index : torch.Tensor
            ``[2, E]`` connectivity.
        batch : torch.Tensor
            ``[N]`` graph assignment (single graph here).

        Returns
        -------
        tuple(list[float], list[float])
            ``(node_mask, edge_mask)`` importance values.
        """
        model_config = ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="log_probs",  # CHEM GAT ends in log_softmax
        )
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=self.epochs),
            explanation_type="model",
            node_mask_type="object",  # one scalar importance per node
            edge_mask_type="object",  # one scalar importance per edge
            model_config=model_config,
        )
        # The Explainer forwards **kwargs to the model; CHEM's forward signature
        # is ``forward(x, edge_index, batch)`` so we pass ``batch`` as a kwarg.
        explanation = explainer(x, edge_index, batch=batch)

        node_mask_t = getattr(explanation, "node_mask", None)
        edge_mask_t = getattr(explanation, "edge_mask", None)
        node_mask = (
            node_mask_t.reshape(x.shape[0], -1).sum(dim=1).detach().cpu().tolist()
            if node_mask_t is not None
            else [0.0] * x.shape[0]
        )
        edge_mask = (
            edge_mask_t.detach().cpu().reshape(-1).tolist()
            if edge_mask_t is not None
            else [0.0] * edge_index.shape[1]
        )
        return node_mask, edge_mask

    def _run_legacy_api(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """Run explanation via the legacy ``torch_geometric.nn.GNNExplainer``.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, F]`` node features.
        edge_index : torch.Tensor
            ``[2, E]`` connectivity.
        batch : torch.Tensor
            ``[N]`` graph assignment (single graph here).

        Returns
        -------
        tuple(list[float], list[float])
            ``(node_mask, edge_mask)`` importance values.
        """
        explainer = LegacyGNNExplainer(self.model, epochs=self.epochs, return_type="log_prob")
        # Legacy API exposes ``explain_graph`` for graph-level classification.
        node_feat_mask, edge_mask_t = explainer.explain_graph(x, edge_index, batch=batch)
        # ``node_feat_mask`` is a per-feature mask; expand to per-node importance
        # by weighting each node's features by the shared feature mask.
        feat_mask = node_feat_mask.detach().to(x.device)
        node_mask = (x.abs() * feat_mask.unsqueeze(0)).sum(dim=1).detach().cpu().tolist()
        edge_mask = edge_mask_t.detach().cpu().reshape(-1).tolist()
        return node_mask, edge_mask
