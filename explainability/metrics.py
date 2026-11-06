"""Small, dependency-free helpers for judging how trustworthy an explanation is.

Two independent explainers (SHAP node attributions and the GNNExplainer
subgraph) answer related questions.  When they *agree* on which elements matter,
the explanation is more trustworthy; when they disagree, or when the model's
score sits close to the decision threshold, a human should look at the case.
These helpers turn those ideas into numbers the report and API can expose.

Public API
----------
``rank_correlation(a, b) -> float``
``top_k_overlap(a_ids, b_ids) -> float``
``explainer_agreement(shap_result, gnn_result, top_fraction=0.3) -> dict``
``decision_band(harm_score, threshold, margin=0.1) -> str``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _ranks(values: Sequence[float]) -> List[float]:
    """Return average ranks (1-based) for ``values``, handling ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length score lists.

    Returns ``0.0`` when it is undefined (fewer than 2 items, unequal lengths,
    or one list is constant).
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra, rb = _ranks(a), _ranks(b)
    mean_a = sum(ra) / len(ra)
    mean_b = sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    return float(cov / (var_a ** 0.5 * var_b ** 0.5))


def top_k_overlap(a_ids: Sequence[str], b_ids: Sequence[str]) -> float:
    """Jaccard overlap of two id collections (``0.0`` if both are empty)."""
    sa, sb = set(a_ids), set(b_ids)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def explainer_agreement(
    shap_result: Optional[Dict[str, Any]],
    gnn_result: Optional[Dict[str, Any]],
    top_fraction: float = 0.3,
) -> Dict[str, Any]:
    """Measure how well the SHAP and GNNExplainer results agree.

    Parameters
    ----------
    shap_result, gnn_result : dict or None
        Outputs of ``SHAPExplainer.explain`` and ``GNNExplainerWrapper.explain``.
    top_fraction : float
        Fraction of nodes counted as "top" SHAP nodes for the overlap measure.

    Returns
    -------
    dict
        ``{"available", "jaccard", "spearman", "agree", "note"}``.  ``agree`` is
        ``True`` when the overlap is at least 0.5 or the rank correlation is at
        least 0.5.  ``available`` is ``False`` if either explainer did not run
        or the GNN mask was flat (no meaningful subgraph).
    """
    if not isinstance(shap_result, dict) or not isinstance(gnn_result, dict):
        return {"available": False, "jaccard": 0.0, "spearman": 0.0, "agree": False,
                "note": "Both explainers are needed to measure agreement."}
    if not gnn_result.get("available") or gnn_result.get("flat_mask"):
        return {"available": False, "jaccard": 0.0, "spearman": 0.0, "agree": False,
                "note": "GNNExplainer produced no usable subgraph."}

    shap_ids = [str(i) for i in shap_result.get("node_ids", [])]
    shap_scores = [float(s) for s in shap_result.get("node_scores", [])]
    if not shap_ids or len(shap_ids) != len(shap_scores):
        return {"available": False, "jaccard": 0.0, "spearman": 0.0, "agree": False,
                "note": "SHAP result has no usable node scores."}

    k = max(1, int(round(len(shap_ids) * top_fraction)))
    top = sorted(range(len(shap_ids)), key=lambda i: shap_scores[i], reverse=True)[:k]
    shap_top_ids = [shap_ids[i] for i in top]
    gnn_ids = [str(n.get("id", "")) for n in gnn_result.get("important_nodes", [])]
    jaccard = top_k_overlap(shap_top_ids, gnn_ids)

    gnn_mask = [float(v) for v in gnn_result.get("node_mask", [])]
    spearman = rank_correlation(shap_scores, gnn_mask) if len(gnn_mask) == len(shap_scores) else 0.0

    agree = jaccard >= 0.5 or spearman >= 0.5
    return {
        "available": True,
        "jaccard": round(jaccard, 4),
        "spearman": round(spearman, 4),
        "agree": bool(agree),
        "note": "Explainers agree." if agree else "Explainers point at different elements.",
    }


def decision_band(harm_score: Optional[float], threshold: Optional[float], margin: float = 0.1) -> str:
    """Classify a score relative to the decision threshold.

    Returns one of ``"harmful"``, ``"borderline_harmful"``,
    ``"borderline_safe"``, ``"safe"`` or ``"unknown"`` (when either input is
    missing).  "Borderline" means within ``margin`` of the threshold.
    """
    if harm_score is None or threshold is None:
        return "unknown"
    score, thr = float(harm_score), float(threshold)
    if abs(score - thr) < margin:
        return "borderline_harmful" if score >= thr else "borderline_safe"
    return "harmful" if score >= thr else "safe"
