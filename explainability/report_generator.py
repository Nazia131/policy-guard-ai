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

from explainability.metrics import decision_band, explainer_agreement
from utils.config import get_policy_clause  # absolute import per spec

# A SHAP attribution is only called "confirming" when ablating its top nodes
# hurts the prediction at least this much more than ablating random nodes.
_MIN_FAITHFULNESS_MARGIN = 0.1

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
        signed_map: Dict[str, float] = {}
        if isinstance(shap_result, dict):
            signed_map = {
                str(i): float(s)
                for i, s in zip(
                    shap_result.get("node_ids", []), shap_result.get("node_signed_scores", [])
                )
            }
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
                supported_by = [
                    name
                    for name, flag in (("rule", harmful), ("shap", high_attr), ("gnn", in_gnn))
                    if flag
                ]
                selected.append(
                    {
                        "id": node_id,
                        "label": str(node.get("full_label", node.get("label", node_id))),
                        "modality": str(node.get("modality", node.get("type", "unknown"))),
                        "confidence": float(node.get("confidence", 0.0)),
                        "attribution": attribution,
                        "signed_attribution": signed_map.get(node_id, 0.0),
                        "harmful_signal": harmful,
                        "supported_by": supported_by,
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
                        "signed_attribution": signed_map.get(node_id, 0.0),
                        "harmful_signal": False,
                        "supported_by": ["shap"],
                    }
                )

        # Elements backed by more independent signals come first.
        selected.sort(
            key=lambda e: (len(e["supported_by"]), e["attribution"], e["confidence"]),
            reverse=True,
        )
        return selected

    def _mitigating_elements(
        self,
        graph_dict: Dict[str, Any],
        shap_result: Optional[Dict[str, Any]],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Elements whose SHAP contribution points toward *safe*.

        These are the benign parts of the content (e.g. the harmless half of a
        meme) and are what a reviewer needs to see to judge a possible false
        positive.  Returns at most ``limit`` elements with the most negative
        signed attribution, or an empty list if signed scores are unavailable.
        """
        if not isinstance(shap_result, dict):
            return []
        ids = [str(i) for i in shap_result.get("node_ids", [])]
        signed = [float(s) for s in shap_result.get("node_signed_scores", [])]
        if not ids or len(ids) != len(signed):
            return []
        nodes = graph_dict.get("nodes")
        if nodes is None:
            data = graph_dict.get("data")
            nodes = getattr(data, "node_meta", None) if data is not None else None
        meta = {str(n.get("id", f"node_{i}")): n for i, n in enumerate(nodes or [])}
        negatives = sorted(
            ((i, s) for i, s in zip(ids, signed) if s < 0.0), key=lambda t: t[1]
        )[:limit]
        out = []
        for node_id, s in negatives:
            node = meta.get(node_id, {})
            out.append(
                {
                    "id": node_id,
                    "label": str(node.get("full_label", node.get("label", node_id))),
                    "modality": str(node.get("modality", node.get("type", "unknown"))),
                    "signed_attribution": s,
                }
            )
        return out

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

    def _decision_aware_text(
        self,
        band: str,
        clause_label: str,
        element_list_str: str,
        pattern_name: str,
        faithfulness: float,
        faithfulness_margin: float,
        harm_score: Optional[float],
        threshold: Optional[float],
        mitigating: List[Dict[str, Any]],
    ) -> str:
        """Build explanation wording that matches how strongly the model decided."""
        if faithfulness_margin >= _MIN_FAITHFULNESS_MARGIN:
            shap_tail = (
                f"SHAP attribution supports these elements at {faithfulness:.2f} faithfulness."
            )
        else:
            shap_tail = (
                f"SHAP attribution only weakly supports these elements "
                f"(faithfulness {faithfulness:.2f}, little better than removing random elements)."
            )
        score_txt = "numeric score not provided"
        if harm_score is not None and threshold is not None:
            score_txt = f"harm score {float(harm_score):.2f} vs threshold {float(threshold):.2f}"
        mitigating_txt = ""
        if mitigating:
            mitigating_txt = " Elements pointing toward safe: " + "; ".join(
                f"'{m['label']}' ({m['modality']})" for m in mitigating
            ) + "."

        if band == "harmful":
            return (
                f"This content was flagged under {clause_label} ({score_txt}). "
                f"The following elements combined to form a harmful pattern: {element_list_str}. "
                f"This combination matches {pattern_name} in the Harm Pattern Library. "
                f"{shap_tail}{mitigating_txt}"
            )
        if band == "borderline_harmful":
            return (
                f"This content was flagged under {clause_label}, but only narrowly "
                f"({score_txt}); human review is recommended. Elements contributing most: "
                f"{element_list_str}. Closest library pattern: {pattern_name}. "
                f"{shap_tail}{mitigating_txt}"
            )
        if band == "borderline_safe":
            return (
                f"This content was not flagged, but it is close to the threshold ({score_txt}); "
                f"human review is recommended. Elements contributing most toward harm: "
                f"{element_list_str}. Closest library pattern: {pattern_name}. "
                f"{shap_tail}{mitigating_txt}"
            )
        return (
            f"This content was not flagged ({score_txt}). Elements contributing most toward "
            f"harm: {element_list_str}, which were not strong enough to cross the threshold."
            f"{mitigating_txt}"
        )

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        graph_dict: Dict[str, Any],
        shap_result: Optional[Dict[str, Any]],
        gnn_result: Optional[Dict[str, Any]],
        matched_pattern: Any,
        policy_clause: Any,
        *,
        harm_score: Optional[float] = None,
        verdict: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Produce the combined structured + human-readable explanation.

        When ``harm_score`` / ``verdict`` / ``threshold`` are supplied the wording
        is decision-aware: confident harmful, borderline, or not flagged.  Without
        them the original fixed template is used (backward compatible).

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
        faithfulness_margin = 0.0
        if isinstance(shap_result, dict):
            faithfulness = float(shap_result.get("faithfulness", 0.0))
            faithfulness_margin = float(
                shap_result.get("faithfulness_margin", faithfulness)
            )
        faithfulness_str = f"{faithfulness:.2f}"

        mitigating = self._mitigating_elements(graph_dict, shap_result)
        agreement = explainer_agreement(shap_result, gnn_result)

        # --- decision band --------------------------------------------------
        if harm_score is not None and threshold is not None:
            band = decision_band(harm_score, threshold)
        elif verdict is not None:
            band = "harmful" if str(verdict).upper() == "HARMFUL" else "safe"
        else:
            band = "unknown"

        if band == "unknown":
            # Legacy behaviour: the exact spec template.
            explanation = _EXPLANATION_TEMPLATE.format(
                policy_clause=clause_label,
                element_list=element_list_str,
                pattern_name=pattern_name,
                faithfulness=faithfulness_str,
            )
        else:
            explanation = self._decision_aware_text(
                band=band,
                clause_label=clause_label,
                element_list_str=element_list_str,
                pattern_name=pattern_name,
                faithfulness=faithfulness,
                faithfulness_margin=faithfulness_margin,
                harm_score=harm_score,
                threshold=threshold,
                mitigating=mitigating,
            )
            if agreement["available"] and not agreement["agree"]:
                explanation += (
                    " The SHAP and subgraph explainers point at different elements, "
                    "so treat this explanation as low confidence."
                )

        needs_human_review = band.startswith("borderline") or (
            agreement["available"] and not agreement["agree"]
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
            "mitigating_elements": mitigating,
            "faithfulness": faithfulness,
            "faithfulness_margin": faithfulness_margin,
            "decision": {
                "band": band,
                "harm_score": harm_score,
                "threshold": threshold,
                "verdict": verdict,
                "needs_human_review": bool(needs_human_review),
            },
            "agreement": agreement,
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
