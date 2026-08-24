"""Human-readable + structured explanation reports for CHEM decisions.

The :class:`ReportGenerator` fuses the outputs of the two explainers
(:mod:`explainability.shap_explainer` and :mod:`explainability.gnn_explainer`)
with the harm-scorer's matched Harm-Pattern-Library entry and the governing
policy clause into a single artefact that a human reviewer (or an appeals
dashboard) can read.  It produces:

* a **structured** dict — machine-consumable, carrying the flagged elements,
  the responsible subgraph, the matched pattern and the policy clause; and
* a **human-readable** explanation string built from the exact template mandated
  by the CHEM spec, so the wording is stable and auditable.

No harm semantics are hardcoded here: policy-clause text is looked up from
``policy/policy_config.json`` via :mod:`utils.config`, and element labels /
modalities / confidences come from the graph the pipeline already built.

Public API
----------
``ReportGenerator()``
``ReportGenerator.generate(graph_dict, shap_result, gnn_result,
                           matched_pattern, policy_clause) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.config import get_policy_clause  # absolute import per spec

# Exact human-readable template mandated by the CHEM spec.  Kept as a module
# constant so the wording is defined in exactly one place.
_EXPLANATION_TEMPLATE = (
    "This content was flagged under {policy_clause}. The following elements "
    "combined to form a harmful pattern: {element_list}. This cross-modal "
    "combination matches {pattern_name} in the Harm Pattern Library. SHAP "
    "attribution confirms these elements are responsible for the decision at "
    "{faithfulness} faithfulness."
)


class ReportGenerator:
    """Combine explainer outputs, pattern match and policy clause into a report.

    The generator holds no models and no per-request state, so a single instance
    can be reused across the whole API lifetime.
    """

    def __init__(self) -> None:
        """Initialise the (stateless) report generator."""
        print("[CHEM][report_generator] Initialised.")

    # -- internal helpers ---------------------------------------------------

    def _resolve_clause_label(self, policy_clause: Any) -> str:
        """Return a display label for the governing policy clause.

        Parameters
        ----------
        policy_clause : dict or str or None
            Either a clause dict (as returned by
            ``utils.config.get_policy_clause`` / the harm scorer), a bare harm
            category id string, or ``None``.

        Returns
        -------
        str
            A label like ``"Hate Speech (MCS-3.2)"`` when possible, else a
            sensible fallback that never breaks the template.
        """
        clause: Optional[Dict[str, Any]] = None
        if isinstance(policy_clause, dict):
            clause = policy_clause
        elif isinstance(policy_clause, str) and policy_clause:
            # Treat the string as a category id and look up its clause.
            clause = get_policy_clause(policy_clause)

        if clause:
            title = clause.get("title") or clause.get("display_name") or "policy"
            clause_id = clause.get("clause_id", "")
            return f"{title} ({clause_id})".strip() if clause_id else str(title)
        return "an applicable content policy"

    def _node_attribution_map(self, shap_result: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Build an ``id -> attribution`` lookup from a SHAP result.

        Parameters
        ----------
        shap_result : dict or None
            Output of :meth:`SHAPExplainer.explain`.

        Returns
        -------
        dict
            Mapping node id -> attribution magnitude (empty if unavailable).
        """
        if not isinstance(shap_result, dict):
            return {}
        ids = shap_result.get("node_ids", [])
        scores = shap_result.get("node_scores", [])
        return {str(i): float(s) for i, s in zip(ids, scores)}

    def _select_elements(
        self,
        graph_dict: Dict[str, Any],
        shap_result: Optional[Dict[str, Any]],
        gnn_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select the elements that combined to form the harmful pattern.

        A node is included if it carries a harmful signal, OR if either explainer
        flagged it as important (high SHAP attribution / in the GNNExplainer
        subgraph).  This unions rule-based harm flags with learned attribution so
        the report reflects both the "what" and the "why".

        Parameters
        ----------
        graph_dict : dict
            CHEM graph dict with a ``"nodes"`` metadata list.
        shap_result : dict or None
            SHAP explainer output (for attribution scores).
        gnn_result : dict or None
            GNNExplainer output (for the important-node subgraph).

        Returns
        -------
        list[dict]
            Each: ``{"id","label","modality","confidence","attribution",
            "harmful_signal"}``, ordered by descending attribution then
            confidence.
        """
        nodes = graph_dict.get("nodes")
        if nodes is None:
            data = graph_dict.get("data")
            nodes = getattr(data, "node_meta", None) if data is not None else None
        nodes = nodes or []

        attr_map = self._node_attribution_map(shap_result)
        gnn_important_ids = set()
        if isinstance(gnn_result, dict):
            for n in gnn_result.get("important_nodes", []):
                gnn_important_ids.add(str(n.get("id", "")))

        # Determine an attribution cutoff (mean) to identify "high" attribution.
        attr_values = [v for v in attr_map.values()]
        attr_cut = (sum(attr_values) / len(attr_values)) if attr_values else 0.0

        selected: List[Dict[str, Any]] = []
        for i, node in enumerate(nodes):
            node_id = str(node.get("id", f"node_{i}"))
            harmful = bool(node.get("harmful_signal", False))
            attribution = attr_map.get(node_id, 0.0)
            high_attr = attribution > attr_cut and attribution > 0.0
            in_gnn = node_id in gnn_important_ids

            if harmful or high_attr or in_gnn:
                selected.append(
                    {
                        "id": node_id,
                        "label": str(node.get("full_label", node.get("label", node_id))),
                        "modality": str(node.get("modality", node.get("type", "unknown"))),
                        "confidence": float(node.get("confidence", 0.0)),
                        "attribution": attribution,
                        "harmful_signal": harmful,
                    }
                )

        # If nothing was selected (e.g. no metadata), fall back to the top-SHAP
        # nodes so the report is never empty when a decision was made.
        if not selected and attr_map:
            top = sorted(attr_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
            for node_id, attribution in top:
                selected.append(
                    {
                        "id": node_id,
                        "label": node_id,
                        "modality": "unknown",
                        "confidence": 0.0,
                        "attribution": attribution,
                        "harmful_signal": False,
                    }
                )

        selected.sort(key=lambda e: (e["attribution"], e["confidence"]), reverse=True)
        return selected

    def _format_element_list(self, elements: List[Dict[str, Any]]) -> str:
        """Render selected elements as a readable clause for the template.

        Parameters
        ----------
        elements : list[dict]
            Elements from :meth:`_select_elements`.

        Returns
        -------
        str
            e.g. ``"'weapon' (image, confidence 0.82); 'kill you' (text,
            confidence 0.91)"`` or a graceful placeholder when empty.
        """
        if not elements:
            return "no individual elements could be isolated"
        parts = [
            f"'{e['label']}' ({e['modality']}, confidence {e['confidence']:.2f})"
            for e in elements
        ]
        return "; ".join(parts)

    def _pattern_name(self, matched_pattern: Any) -> str:
        """Extract a human name for the matched Harm-Pattern-Library entry.

        Parameters
        ----------
        matched_pattern : dict or str or None
            Pattern entry from the harm scorer, a bare name, or ``None``.

        Returns
        -------
        str
            A pattern name/id, or ``"no specific stored pattern"`` if absent.
        """
        if isinstance(matched_pattern, dict):
            return str(
                matched_pattern.get("name")
                or matched_pattern.get("pattern_id")
                or matched_pattern.get("policy_category")
                or "an unnamed pattern"
            )
        if isinstance(matched_pattern, str) and matched_pattern:
            return matched_pattern
        return "no specific stored pattern"

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        graph_dict: Dict[str, Any],
        shap_result: Optional[Dict[str, Any]],
        gnn_result: Optional[Dict[str, Any]],
        matched_pattern: Any,
        policy_clause: Any,
    ) -> Dict[str, Any]:
        """Produce the combined structured + human-readable explanation.

        Parameters
        ----------
        graph_dict : dict
            The CHEM content-composition graph dict for the flagged content.
        shap_result : dict or None
            Output of :meth:`SHAPExplainer.explain` (node attributions +
            faithfulness).  May be ``None`` if SHAP was skipped.
        gnn_result : dict or None
            Output of :meth:`GNNExplainerWrapper.explain` (responsible subgraph).
            May be ``None`` if it was skipped.
        matched_pattern : dict or str or None
            The Harm-Pattern-Library entry the content matched (from the harm
            scorer), or ``None`` if no stored pattern matched.
        policy_clause : dict or str or None
            The governing policy clause dict or a harm-category id.

        Returns
        -------
        dict
            ``{
                "explanation": str,      # exact-template human-readable string
                "structured": {          # machine-consumable detail
                    "policy_clause": {...},
                    "matched_pattern": ...,
                    "pattern_name": str,
                    "elements": [ {id,label,modality,confidence,attribution,
                                   harmful_signal}, ... ],
                    "faithfulness": float,
                    "shap": {available, method, note},
                    "gnn": {available, backend, important_nodes, important_edges},
                }
            }``
            Never raises; missing inputs degrade to safe placeholders.
        """
        # --- resolve pieces ------------------------------------------------
        clause_label = self._resolve_clause_label(policy_clause)
        elements = self._select_elements(graph_dict, shap_result, gnn_result)
        element_list_str = self._format_element_list(elements)
        pattern_name = self._pattern_name(matched_pattern)

        faithfulness = 0.0
        if isinstance(shap_result, dict):
            faithfulness = float(shap_result.get("faithfulness", 0.0))
        faithfulness_str = f"{faithfulness:.2f}"

        # --- human-readable string (EXACT template) ------------------------
        explanation = _EXPLANATION_TEMPLATE.format(
            policy_clause=clause_label,
            element_list=element_list_str,
            pattern_name=pattern_name,
            faithfulness=faithfulness_str,
        )

        # --- structured payload -------------------------------------------
        clause_struct: Any = policy_clause
        if isinstance(policy_clause, str) and policy_clause:
            clause_struct = get_policy_clause(policy_clause)

        structured: Dict[str, Any] = {
            "policy_clause": clause_struct,
            "policy_clause_label": clause_label,
            "matched_pattern": matched_pattern,
            "pattern_name": pattern_name,
            "elements": elements,
            "faithfulness": faithfulness,
            "shap": {
                "available": bool(shap_result.get("available")) if isinstance(shap_result, dict) else False,
                "method": shap_result.get("method") if isinstance(shap_result, dict) else None,
                "note": shap_result.get("note") if isinstance(shap_result, dict) else "SHAP not run",
            },
            "gnn": {
                "available": bool(gnn_result.get("available")) if isinstance(gnn_result, dict) else False,
                "backend": gnn_result.get("backend") if isinstance(gnn_result, dict) else None,
                "important_nodes": gnn_result.get("important_nodes", []) if isinstance(gnn_result, dict) else [],
                "important_edges": gnn_result.get("important_edges", []) if isinstance(gnn_result, dict) else [],
            },
        }

        print(
            f"[CHEM][report_generator] Generated report: {len(elements)} elements, "
            f"pattern='{pattern_name}', faithfulness={faithfulness_str}."
        )
        return {"explanation": explanation, "structured": structured}
