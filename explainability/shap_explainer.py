"""SHAP-based node attribution for the CHEM Graph Attention Network.

This module explains an individual harm decision produced by
:class:`model.gat_model.CHEMGraphAttentionNetwork` by attributing the model's
output back to the *node features* of the content-composition graph.  Each node
in a CHEM graph is one extracted content element (a piece of text, an image
region, an audio segment, a detected action, ...), so a per-node attribution
tells a human reviewer *which elements* drove the harmful classification.

Approach
--------
SHAP's :class:`shap.GradientExplainer` expects a callable that maps a batch of
input tensors to a batch of scalar/vector outputs, together with a small
*background* set of representative inputs.  A graph neural network does not fit
that signature directly (its forward pass also needs ``edge_index`` and a
``batch`` vector), so we wrap the GAT in a small ``nn.Module`` that:

* freezes the current graph topology (``edge_index`` / ``batch``) for the graph
  being explained, and
* exposes a ``forward(x)`` that takes only the ``[N, 768]`` node-feature matrix
  and returns the harmful-class log/soft probability.

We build a background set by sampling perturbations of the graph's own node
features (Gaussian jitter + a zero baseline).  GradientExplainer then integrates
gradients between each background sample and the real node features, yielding a
``[N, 768]`` SHAP value tensor.  We collapse the feature dimension (L2 / sum of
absolute values) into a single scalar attribution per node.

Faithfulness
------------
An explanation is only useful if the highlighted nodes are genuinely
responsible for the prediction.  We estimate a ``faithfulness`` score in
``[0, 1]`` by *ablating* the top-attributed nodes (zeroing their features) and
measuring how much the harmful probability drops, normalised against the
original probability.  A large drop when removing the "important" nodes means
the attribution is faithful.  When SHAP is unavailable we still return a
gradient-based fallback so downstream code never crashes.

Public API
----------
``SHAPExplainer(model, device=None)``
``SHAPExplainer.explain(graph_dict) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# SHAP is a runtime dependency; it is intentionally imported at module top so the
# dependency is declared.  It is not installed in every environment (e.g. static
# analysis / CI), so we degrade gracefully instead of failing at import time.
try:  # pragma: no cover - exercised only when the dependency is present
    import shap  # type: ignore

    _SHAP_AVAILABLE = True
    _SHAP_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - dependency missing
    shap = None  # type: ignore
    _SHAP_AVAILABLE = False
    _SHAP_IMPORT_ERROR = _exc

from model.gat_model import CHEMGraphAttentionNetwork  # noqa: E402  (absolute import per spec)


class _NodeFeatureModelWrapper(nn.Module):
    """Adapt a CHEM GAT to a ``forward(x) -> harmful_prob`` signature for SHAP.

    SHAP's explainers operate on a single input tensor, but a graph network's
    forward pass also needs the edge structure and a batch assignment.  This
    wrapper *closes over* the fixed topology of one graph so that only the node
    feature matrix ``x`` varies.  This is exactly what we want: SHAP perturbs the
    node features and observes the change in harmful probability while the graph
    connectivity stays constant.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The trained GAT to explain.  Kept in eval mode.
    edge_index : torch.Tensor
        ``[2, E]`` long tensor describing the (fixed) graph connectivity.
    batch : torch.Tensor
        ``[N]`` long tensor assigning every node to graph 0 (single graph).
    harmful_index : int
        Column of the model output corresponding to the "harmful" class.

    Returns (from ``forward``)
    --------------------------
    torch.Tensor
        ``[B, 1]`` harmful-class probabilities, one per input feature matrix in
        the batch dimension that SHAP supplies.
    """

    def __init__(
        self,
        model: CHEMGraphAttentionNetwork,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        harmful_index: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        # Register topology as buffers so device moves keep them aligned with x.
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_batch", batch)
        self.harmful_index = harmful_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GAT on ``x`` under the frozen topology and return harm prob.

        Parameters
        ----------
        x : torch.Tensor
            Either ``[N, F]`` (single graph) or ``[B, N, F]`` (SHAP may pass a
            batch dimension).  Each ``[N, F]`` slice is scored independently.

        Returns
        -------
        torch.Tensor
            ``[B, 1]`` harmful-class probabilities.
        """
        if x.dim() == 2:
            # Single graph: add a batch dimension for uniform handling.
            x = x.unsqueeze(0)

        outputs: List[torch.Tensor] = []
        for i in range(x.shape[0]):
            node_features = x[i]
            # The GAT returns log_softmax over [safe, harmful]; exponentiate to
            # recover a probability in [0, 1].
            log_probs = self.model(node_features, self._edge_index, self._batch)
            probs = torch.exp(log_probs)
            # ``probs`` is [1, 2] (one pooled graph); pick the harmful column.
            harmful = probs[:, self.harmful_index]
            outputs.append(harmful)
        return torch.stack(outputs, dim=0)  # [B, 1]


class SHAPExplainer:
    """Explain a CHEM harm decision with SHAP node-feature attributions.

    The explainer wraps a trained :class:`CHEMGraphAttentionNetwork` and, for a
    given graph, produces a per-node attribution score plus a faithfulness
    estimate.  Attribution magnitude answers *"how much did this element push
    the model toward a harmful verdict?"*.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        A trained GAT.  It is put in eval mode and never trained here.
    device : str or torch.device, optional
        Device to run explanation on.  Defaults to CUDA if available else CPU.
    """

    def __init__(
        self,
        model: CHEMGraphAttentionNetwork,
        device: Optional[Any] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self._shap_available = _SHAP_AVAILABLE
        print(
            f"[CHEM][shap_explainer] Initialised (device={self.device}, "
            f"shap_available={self._shap_available})"
        )

    # -- internal helpers ---------------------------------------------------

    def _extract_tensors(self, graph_dict: Dict[str, Any]):
        """Pull node features, edge index and node metadata from a graph dict.

        Parameters
        ----------
        graph_dict : dict
            Output of ``ContentCompositionGraphBuilder.build`` containing a
            ``torch_geometric.data.Data`` under ``"data"`` (with ``x`` and
            ``edge_index``) and a ``"nodes"`` list of JSON-serialisable metadata.

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor, list)
            ``(x [N, F], edge_index [2, E], batch [N], node_meta list)``.

        Raises
        ------
        KeyError
            If the graph dict does not contain a usable ``data`` object.
        """
        data = graph_dict.get("data")
        if data is None:
            raise KeyError("graph_dict must contain a 'data' PyG object with x/edge_index")

        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device).long()
        num_nodes = x.shape[0]
        batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        # Node metadata is used to label attributions; fall back to the graph's
        # serialisable node list, then to synthetic ids.
        node_meta = getattr(data, "node_meta", None)
        if node_meta is None:
            node_meta = graph_dict.get("nodes")
        if node_meta is None:
            node_meta = [{"id": f"node_{i}"} for i in range(num_nodes)]
        return x, edge_index, batch, node_meta

    def _build_background(self, x: torch.Tensor, num_samples: int = 16) -> torch.Tensor:
        """Construct a small SHAP background set from the graph's own features.

        A good background represents "typical / uninformative" inputs.  For a
        single graph we synthesise one by (a) including an all-zero baseline and
        (b) adding Gaussian jitter scaled to each feature's own std, producing
        plausible neighbouring node-feature matrices.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, F]`` node feature matrix for the graph being explained.
        num_samples : int
            Number of background matrices to generate (kept small for speed).

        Returns
        -------
        torch.Tensor
            ``[num_samples, N, F]`` background tensor.
        """
        feature_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        samples = [torch.zeros_like(x)]  # zero baseline
        for _ in range(max(1, num_samples - 1)):
            noise = torch.randn_like(x) * feature_std * 0.5
            samples.append(x + noise)
        return torch.stack(samples, dim=0)

    def _faithfulness(
        self,
        wrapper: _NodeFeatureModelWrapper,
        x: torch.Tensor,
        node_scores: List[float],
        top_fraction: float = 0.3,
    ) -> float:
        """Estimate attribution faithfulness by ablating top-attributed nodes.

        We zero the features of the highest-attributed nodes and measure the
        resulting drop in harmful probability, normalised by the original
        probability.  A faithful explanation concentrates responsibility on the
        nodes it highlights, so removing them collapses the harmful score.

        Parameters
        ----------
        wrapper : _NodeFeatureModelWrapper
            Callable mapping ``[N, F]`` node features to harmful probability.
        x : torch.Tensor
            ``[N, F]`` original node features.
        node_scores : list[float]
            Per-node attribution magnitudes (aligned to rows of ``x``).
        top_fraction : float
            Fraction of nodes (by attribution) to ablate.

        Returns
        -------
        float
            Faithfulness in ``[0, 1]`` (relative probability drop, clamped).
        """
        with torch.no_grad():
            base_prob = float(wrapper(x).flatten()[0].item())
            if base_prob <= 1e-6:
                # Nothing to explain (model already ~safe); treat as neutral.
                return 0.0

            num_nodes = x.shape[0]
            k = max(1, int(round(num_nodes * top_fraction)))
            order = sorted(range(num_nodes), key=lambda i: node_scores[i], reverse=True)
            ablate_idx = order[:k]

            x_ablated = x.clone()
            x_ablated[ablate_idx] = 0.0
            ablated_prob = float(wrapper(x_ablated).flatten()[0].item())

            drop = (base_prob - ablated_prob) / base_prob
            return float(max(0.0, min(1.0, drop)))

    def _gradient_fallback(
        self, wrapper: _NodeFeatureModelWrapper, x: torch.Tensor
    ) -> List[float]:
        """Compute a plain input-gradient saliency when SHAP is unavailable.

        Uses ``grad(harmful_prob, x)`` and collapses the feature dimension by L2
        norm to produce one attribution magnitude per node.  This mirrors what
        SHAP would report (roughly ``gradient * input``) but needs no external
        dependency, so ``explain`` still returns useful per-node scores.

        Parameters
        ----------
        wrapper : _NodeFeatureModelWrapper
            The frozen-topology model wrapper.
        x : torch.Tensor
            ``[N, F]`` node features.

        Returns
        -------
        list[float]
            Per-node attribution magnitudes.
        """
        x_req = x.clone().detach().requires_grad_(True)
        harmful = wrapper(x_req).flatten()[0]
        harmful.backward()
        grads = x_req.grad if x_req.grad is not None else torch.zeros_like(x_req)
        # gradient * input, then per-node L2 norm across features.
        contribution = (grads * x_req).detach()
        node_scores = contribution.norm(dim=1)
        return [float(v) for v in node_scores.cpu().tolist()]

    # -- public API ---------------------------------------------------------

    def explain(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Produce per-node SHAP attributions and a faithfulness score.

        Parameters
        ----------
        graph_dict : dict
            A CHEM graph dict from ``ContentCompositionGraphBuilder.build`` with
            a ``"data"`` PyG object and ``"nodes"`` metadata.

        Returns
        -------
        dict
            ``{
                "node_scores":   list[float],   # attribution magnitude per node
                "node_ids":      list[str],     # aligned node identifiers
                "node_labels":   list[str],     # aligned human labels
                "faithfulness":  float,         # in [0, 1]
                "method":        str,           # "shap_gradient" | "gradient_fallback"
                "available":     bool,          # whether SHAP itself ran
                "note":          str,           # human-readable status
            }``
            On unrecoverable error the same keys are present with empty/zero
            values and an explanatory ``note`` (never raises to the caller).
        """
        try:
            x, edge_index, batch, node_meta = self._extract_tensors(graph_dict)
        except Exception as exc:  # graph is malformed -> degrade, do not crash
            print(f"[CHEM][shap_explainer] Could not read graph: {exc}")
            return {
                "node_scores": [],
                "node_ids": [],
                "node_labels": [],
                "faithfulness": 0.0,
                "method": "unavailable",
                "available": False,
                "note": f"Invalid graph_dict: {exc}",
            }

        num_nodes = x.shape[0]
        node_ids = [str(m.get("id", f"node_{i}")) for i, m in enumerate(node_meta)]
        node_labels = [
            str(m.get("full_label", m.get("label", node_ids[i])))
            for i, m in enumerate(node_meta)
        ]

        wrapper = _NodeFeatureModelWrapper(self.model, edge_index, batch).to(self.device)
        wrapper.eval()

        node_scores: List[float]
        method: str
        note: str

        if self._shap_available:
            try:
                print("[CHEM][shap_explainer] Running SHAP GradientExplainer ...")
                background = self._build_background(x)
                explainer = shap.GradientExplainer(wrapper, background)
                # SHAP expects the input in the same shape family as background.
                shap_values = explainer.shap_values(x.unsqueeze(0))
                # ``shap_values`` may be a list (one per output) or an array.
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                tensor_vals = torch.as_tensor(values, dtype=torch.float32)
                # Collapse any leading batch dim and the feature dim -> [N].
                tensor_vals = tensor_vals.reshape(-1, num_nodes, x.shape[1])[0]
                node_scores = [float(v) for v in tensor_vals.abs().sum(dim=1).tolist()]
                method = "shap_gradient"
                note = "SHAP GradientExplainer attributions (|value| summed over features)."
            except Exception as exc:  # SHAP present but failed -> fall back
                print(f"[CHEM][shap_explainer] SHAP failed ({exc}); using gradient fallback.")
                node_scores = self._gradient_fallback(wrapper, x)
                method = "gradient_fallback"
                note = f"SHAP raised '{exc}'; used input-gradient saliency instead."
        else:
            print(
                "[CHEM][shap_explainer] SHAP not installed "
                f"({_SHAP_IMPORT_ERROR}); using gradient fallback."
            )
            node_scores = self._gradient_fallback(wrapper, x)
            method = "gradient_fallback"
            note = "SHAP dependency unavailable; used input-gradient saliency instead."

        faithfulness = self._faithfulness(wrapper, x, node_scores)

        print(
            f"[CHEM][shap_explainer] Done: {num_nodes} nodes, method={method}, "
            f"faithfulness={faithfulness:.3f}"
        )
        return {
            "node_scores": node_scores,
            "node_ids": node_ids,
            "node_labels": node_labels,
            "faithfulness": faithfulness,
            "method": method,
            "available": self._shap_available and method == "shap_gradient",
            "note": note,
        }
