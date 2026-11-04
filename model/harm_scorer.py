"""CHEM harm scorer -- inference-time fusion of GAT + pattern library.

This module turns a built Content Composition Graph into a final harm verdict.
It combines two complementary signals:

1. **Learned signal** -- the trained :class:`CHEMGraphAttentionNetwork` predicts
   a harmful-class probability for the whole graph.
2. **Structural signal** -- the graph is compared against a library of mined
   *harmful subgraph patterns* (each pre-tagged with the policy clause it
   violates).  Similarity to the closest pattern both boosts the score and
   supplies a human-readable policy clause / matched pattern for the report.

The two are blended with weights read from ``policy_config.json``
(``gat_vs_pattern_weight`` and ``pattern_similarity_weight``) and thresholded at
``final_harm_threshold`` -- no harm numbers are hardcoded here.

The GAT weights and the pattern library are each loaded **once** in
:meth:`HarmScorer.__init__` and reused for every :meth:`HarmScorer.score` call.

This module also exposes :func:`fuse_modality_scores`, the dynamic cross-modal
weight-normalisation used by the API pipeline when multiple modalities are
present in one piece of content.

Public API
----------
``HarmScorer(model_path, library_path=None, device=None)``
``HarmScorer.score(graph_dict) -> dict``
``fuse_modality_scores(modality_scores: dict) -> dict``
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch

from model.gat_model import CHEMGraphAttentionNetwork
from utils.config import get_coefficient, get_policy_clause

try:  # networkx powers the graph-edit-distance structural comparison.
    import networkx as nx
    from networkx.readwrite import json_graph
except Exception:  # pragma: no cover - degrade gracefully if unavailable
    nx = None
    json_graph = None


# Default location of the mined harm-pattern library (see harm_pattern_library/).
_DEFAULT_LIBRARY_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "harm_pattern_library", "library.json"
    )
)

# Wall-clock budget (seconds) for each graph_edit_distance comparison.  GED is
# NP-hard, so we cap it and fall back to a cheap graph-kernel similarity if the
# optimiser does not converge in time.
_GED_TIMEOUT_SECONDS = 2.0


class HarmScorer:
    """Score a Content Composition Graph and produce a policy-grounded verdict.

    Parameters
    ----------
    model_path : str
        Path to a saved GAT checkpoint (a ``state_dict`` produced by
        ``model/train.py``).
    library_path : str, optional
        Path to the mined harm-pattern library JSON.  Defaults to
        ``harm_pattern_library/library.json``.  A missing library degrades
        gracefully to a GAT-only score.
    device : str, optional
        Torch device string.  Auto-selected (CUDA if available, else CPU).

    Attributes
    ----------
    model : CHEMGraphAttentionNetwork
        The loaded, eval-mode GAT.
    patterns : list[dict]
        The loaded harm-pattern library entries (possibly empty).
    device : torch.device
        Resolved compute device.
    """

    def __init__(
        self,
        model_path: str,
        library_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Load the GAT weights and the harm-pattern library exactly once.

        Inputs: checkpoint path, optional library path, optional device.
        Outputs: a ready :class:`HarmScorer` with model in eval mode.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Load the GAT once. --------------------------------------------
        print(f"[CHEM][harm_scorer] Loading GAT weights from: {model_path}")
        self.model = CHEMGraphAttentionNetwork().to(self.device)
        if model_path and os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Accept either a bare state_dict or a wrapped {"state_dict": ...}.
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(
                checkpoint, dict
            ) else checkpoint
            self.model.load_state_dict(state_dict)
            print("[CHEM][harm_scorer] GAT weights loaded.")
        else:
            print(
                "[CHEM][harm_scorer] WARNING: checkpoint not found at "
                f"{model_path!r}; using randomly-initialised GAT."
            )
        self.model.eval()

        # --- Load the pattern library once. --------------------------------
        self.library_path = library_path or _DEFAULT_LIBRARY_PATH
        self.patterns = self._load_library(self.library_path)
        print(
            f"[CHEM][harm_scorer] Loaded {len(self.patterns)} harm pattern(s) "
            f"from {self.library_path}"
        )

        # --- Cache fusion coefficients from the policy config. -------------
        self.gat_weight = get_coefficient("gat_vs_pattern_weight")
        self.pattern_weight = get_coefficient("pattern_similarity_weight")
        self.final_threshold = get_coefficient("final_harm_threshold")

    # ------------------------------------------------------------------ #
    # Library loading + parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_library(path: str) -> List[Dict[str, Any]]:
        """Load and normalise the harm-pattern library JSON.

        Parameters
        ----------
        path : str
            Path to ``library.json``.

        Returns
        -------
        list[dict]
            Library entries; an empty list if the file is missing or malformed
            (graceful degradation to a GAT-only score).
        """
        if not path or not os.path.isfile(path):
            print(
                f"[CHEM][harm_scorer] WARNING: pattern library not found at "
                f"{path!r}; scoring will use the GAT signal only."
            )
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"[CHEM][harm_scorer] WARNING: failed to read library: {exc}")
            return []
        # The library may be a bare list or wrapped under a "patterns" key.
        if isinstance(data, dict):
            return list(data.get("patterns", []))
        return list(data)

    @staticmethod
    def _pattern_to_nx(pattern: Dict[str, Any]):
        """Rebuild a NetworkX graph from a library pattern's node-link JSON.

        Parameters
        ----------
        pattern : dict
            A library entry whose ``"graph"`` field is a node-link dict.

        Returns
        -------
        networkx.Graph or None
            The reconstructed graph, or ``None`` if it cannot be parsed / nx is
            unavailable.
        """
        if json_graph is None:
            return None
        graph_json = pattern.get("graph")
        if not graph_json:
            return None
        try:
            return json_graph.node_link_graph(graph_json)
        except Exception:  # pragma: no cover - tolerate odd serialisations
            return None

    # ------------------------------------------------------------------ #
    # Structural similarity
    # ------------------------------------------------------------------ #
    def _graph_similarity(self, content_graph, pattern_graph) -> float:
        """Return a similarity in ``[0, 1]`` between two graphs.

        Uses ``networkx.graph_edit_distance`` with a timeout; if that does not
        converge (GED is NP-hard) it falls back to a cheap size/label graph
        kernel.  Distance is converted to similarity via ``1/(1+distance)``,
        normalised by the combined graph size so larger graphs are not unfairly
        penalised.

        Parameters
        ----------
        content_graph, pattern_graph : networkx.Graph
            The graph under test and a library pattern graph.

        Returns
        -------
        float
            Similarity score in ``[0, 1]`` (1.0 == structurally identical).
        """
        if nx is None or content_graph is None or pattern_graph is None:
            return 0.0

        # Node/edge label matchers so GED compares *semantics*, not just shape.
        def node_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            """Return True when two nodes share the same type and harmful-signal flag."""
            return a.get("type") == b.get("type") and a.get(
                "harmful_signal"
            ) == b.get("harmful_signal")

        def edge_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            """Return True when two edges share the same semantic edge-type."""
            return a.get("edge_type") == b.get("edge_type")

        distance: Optional[float] = None
        try:
            distance = nx.graph_edit_distance(
                content_graph,
                pattern_graph,
                node_match=node_match,
                edge_match=edge_match,
                timeout=_GED_TIMEOUT_SECONDS,
            )
        except Exception:  # pragma: no cover - any GED failure -> fallback
            distance = None

        if distance is None:
            # Graph-kernel fallback: label-multiset Jaccard over nodes+edges.
            return self._kernel_similarity(content_graph, pattern_graph)

        # Normalise distance by the larger graph so similarity is scale-aware.
        denom = max(
            content_graph.number_of_nodes()
            + content_graph.number_of_edges()
            + pattern_graph.number_of_nodes()
            + pattern_graph.number_of_edges(),
            1,
        )
        normalised = distance / denom
        return float(max(0.0, min(1.0, 1.0 - normalised)))

    @staticmethod
    def _kernel_similarity(content_graph, pattern_graph) -> float:
        """Cheap label-multiset Jaccard similarity used when GED times out.

        Parameters
        ----------
        content_graph, pattern_graph : networkx.Graph
            Graphs to compare.

        Returns
        -------
        float
            Jaccard overlap of node-type and edge-type label multisets, in
            ``[0, 1]``.
        """
        def node_labels(g) -> List[str]:
            """Return sorted list of node-type strings for a graph."""
            return sorted(str(d.get("type")) for _, d in g.nodes(data=True))

        def edge_labels(g) -> List[str]:
            """Return sorted list of edge-type strings for a graph."""
            return sorted(str(d.get("edge_type")) for *_e, d in g.edges(data=True))

        a = set(node_labels(content_graph)) | {
            f"e:{lbl}" for lbl in edge_labels(content_graph)
        }
        b = set(node_labels(pattern_graph)) | {
            f"e:{lbl}" for lbl in edge_labels(pattern_graph)
        }
        if not a and not b:
            return 0.0
        return float(len(a & b) / max(len(a | b), 1))

    def _best_pattern(self, content_graph) -> Dict[str, Any]:
        """Find the closest-matching library pattern to the content graph.

        Parameters
        ----------
        content_graph : networkx.Graph or None
            The graph under test.

        Returns
        -------
        dict
            ``{"similarity": float, "pattern": dict|None,
               "policy_category": str|None, "policy_clause": dict|None}``.
        """
        best = {
            "similarity": 0.0,
            "pattern": None,
            "policy_category": None,
            "policy_clause": None,
        }
        if content_graph is None:
            return best

        for pattern in self.patterns:
            pattern_graph = self._pattern_to_nx(pattern)
            sim = self._graph_similarity(content_graph, pattern_graph)
            if sim > best["similarity"]:
                category = pattern.get("policy_category")
                best = {
                    "similarity": sim,
                    "pattern": pattern,
                    "policy_category": category,
                    "policy_clause": pattern.get("policy_clause")
                    or (get_policy_clause(category) if category else None),
                }
        return best

    # ------------------------------------------------------------------ #
    # GAT inference
    # ------------------------------------------------------------------ #
    def _gat_harm_probability(self, data) -> tuple:
        """Run the GAT and return the harmful-class probability + attention.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The built graph (needs ``x`` and ``edge_index``).

        Returns
        -------
        tuple(float, dict)
            ``(harm_prob, attention_weights)`` where ``harm_prob`` is the
            class-1 probability (``exp`` of ``log_softmax``) and
            ``attention_weights`` is the dict from
            :meth:`CHEMGraphAttentionNetwork.get_attention_weights`, converted
            to plain Python lists so it is JSON-serialisable.
        """
        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device).long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            attn = self.model.get_attention_weights(x, edge_index, batch)
            log_probs = attn["log_probs"]
            probs = torch.exp(log_probs)  # -> probabilities
            # class 1 == HARMFUL; take the first graph in the (single-graph) batch.
            harm_prob = float(probs[0, 1].item())

        serialisable_attn = {
            "layer1": {
                "edge_index": attn["layer1"]["edge_index"].cpu().tolist(),
                "alpha": attn["layer1"]["alpha"].cpu().tolist(),
            },
            "layer2": {
                "edge_index": attn["layer2"]["edge_index"].cpu().tolist(),
                "alpha": attn["layer2"]["alpha"].cpu().tolist(),
            },
        }
        return harm_prob, serialisable_attn

    # ------------------------------------------------------------------ #
    # Public scoring entry point
    # ------------------------------------------------------------------ #
    def score(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a final harm score, verdict and explanation for one graph.

        Parameters
        ----------
        graph_dict : dict
            The output of ``graph.graph_builder.build_graph`` / ``build`` --
            expects at least ``"data"`` (a PyG ``Data``), ``"networkx"`` (an
            ``nx`` graph) and ``"nodes"`` (JSON-serialisable node dicts).

        Returns
        -------
        dict
            ``{"harm_score": float, "verdict": "HARMFUL"|"SAFE",
               "matched_pattern": dict|None, "policy_clause": dict|None,
               "triggered_elements": list[dict], "attention_weights": dict}``.
        """
        print("[CHEM][harm_scorer] Scoring content composition graph ...")

        # --- 1. Learned GAT signal. ----------------------------------------
        data = graph_dict.get("data")
        if data is None:
            raise ValueError(
                "graph_dict must contain a 'data' (torch_geometric.data.Data) entry."
            )
        gat_prob, attention_weights = self._gat_harm_probability(data)
        print(f"[CHEM][harm_scorer] GAT harmful probability = {gat_prob:.4f}")

        # --- 2. Structural pattern-library signal. -------------------------
        content_graph = graph_dict.get("networkx")
        best = self._best_pattern(content_graph)
        best_sim = float(best["similarity"])
        print(
            f"[CHEM][harm_scorer] Best pattern similarity = {best_sim:.4f} "
            f"(category={best['policy_category']})"
        )

        # --- 3. Weighted fusion + threshold. -------------------------------
        final_score = self.gat_weight * gat_prob + self.pattern_weight * best_sim
        final_score = float(max(0.0, min(1.0, final_score)))
        verdict = "HARMFUL" if final_score >= self.final_threshold else "SAFE"
        print(
            f"[CHEM][harm_scorer] Final harm score = {final_score:.4f} "
            f"-> verdict = {verdict}"
        )

        # --- 4. Triggered elements = the harmful-signal nodes. -------------
        triggered_elements = [
            node
            for node in graph_dict.get("nodes", [])
            if node.get("harmful_signal")
        ]

        # Prefer the matched pattern's clause; else derive from a triggered
        # element's policy category; else leave unset.
        policy_clause = best["policy_clause"]
        if policy_clause is None and triggered_elements:
            category = triggered_elements[0].get("policy_category")
            if category:
                policy_clause = get_policy_clause(category)

        return {
            "harm_score": final_score,
            "verdict": verdict,
            "matched_pattern": best["pattern"],
            "policy_clause": policy_clause,
            "triggered_elements": triggered_elements,
            "attention_weights": attention_weights,
        }


# ---------------------------------------------------------------------------
# Cross-modal fusion (module-level helper).
# ---------------------------------------------------------------------------
def fuse_modality_scores(modality_scores: Dict[str, float]) -> Dict[str, Any]:
    """Dynamically fuse per-modality harm scores into one cross-modal score.

    CHEM does not use fixed modality weights.  Instead it normalises equal
    weight over whichever modalities are *present* in the content, then adds a
    ``cross_modal_bonus`` when a harmful signal is corroborated across at least
    two modalities (an "emergent" cross-modal harm).  Image counts as a harmful
    corroborating modality only if its score clears ``cross_modal_min_image_score``
    and audio only if it clears ``cross_modal_min_audio_score`` -- all read from
    ``policy_config.json``.

    Parameters
    ----------
    modality_scores : dict
        Mapping of modality name (``"text"``/``"image"``/``"audio"``/``"video"``)
        to its scalar harm score in ``[0, 1]``.  Absent modalities are simply
        omitted from the dict.

    Returns
    -------
    dict
        ``{"fused_score": float, "cross_modal_flag": bool}`` where ``fused_score``
        is clamped to ``[0, 1]`` and ``cross_modal_flag`` is ``True`` when the
        cross-modal bonus was applied.
    """
    # Read tunable coefficients from the policy config (never hardcode).
    cross_modal_bonus = get_coefficient("cross_modal_bonus")
    min_image = get_coefficient("cross_modal_min_image_score")
    min_audio = get_coefficient("cross_modal_min_audio_score")

    # Only consider modalities that are actually present (non-None).
    present = {
        modality: float(score)
        for modality, score in (modality_scores or {}).items()
        if score is not None
    }
    if not present:
        return {"fused_score": 0.0, "cross_modal_flag": False}

    # Equal, dynamically-normalised weight over the present modalities.
    weight = 1.0 / len(present)
    base_score = sum(weight * score for score in present.values())

    # Count how many modalities carry a *credible* harmful signal.  For image
    # and audio a per-modality minimum-confidence gate applies; text/video use
    # a strictly-positive score as the presence-of-signal test.
    harmful_modalities = 0
    for modality, score in present.items():
        if modality == "image":
            if score >= min_image:
                harmful_modalities += 1
        elif modality == "audio":
            if score >= min_audio:
                harmful_modalities += 1
        else:
            if score > 0.0:
                harmful_modalities += 1

    cross_modal_flag = harmful_modalities >= 2
    fused_score = base_score + (cross_modal_bonus if cross_modal_flag else 0.0)
    fused_score = float(max(0.0, min(1.0, fused_score)))

    return {"fused_score": fused_score, "cross_modal_flag": cross_modal_flag}
