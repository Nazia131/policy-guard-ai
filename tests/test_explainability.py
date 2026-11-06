"""Tests for the explainability layer: reproducibility, sign, agreement, wording."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from explainability.gnn_explainer import GNNExplainerWrapper  # noqa: E402
from explainability.metrics import (  # noqa: E402
    decision_band,
    explainer_agreement,
    rank_correlation,
    top_k_overlap,
)
from explainability.report_generator import ReportGenerator  # noqa: E402
from explainability.shap_explainer import SHAPExplainer  # noqa: E402
from model.gat_model import CHEMGraphAttentionNetwork  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CHEMGraphAttentionNetwork().eval()


@pytest.fixture()
def graph():
    g = torch.Generator().manual_seed(1)
    n = 6
    x = torch.randn(n, 768, generator=g)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 0, 0, 1]])
    return {
        "data": Data(x=x, edge_index=edge_index),
        "nodes": [
            {
                "id": f"n{i}",
                "label": f"el{i}",
                "modality": "text" if i % 2 == 0 else "image",
                "confidence": 0.5,
                "harmful_signal": i == 0,
            }
            for i in range(n)
        ],
    }


# --- metrics ----------------------------------------------------------------


def test_rank_correlation_known_values():
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rank_correlation([1, 1, 1], [1, 2, 3]) == 0.0
    assert rank_correlation([1], [1]) == 0.0


def test_top_k_overlap():
    assert top_k_overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert top_k_overlap([], []) == 0.0


def test_decision_band():
    assert decision_band(0.9, 0.5) == "harmful"
    assert decision_band(0.55, 0.5) == "borderline_harmful"
    assert decision_band(0.45, 0.5) == "borderline_safe"
    assert decision_band(0.1, 0.5) == "safe"
    assert decision_band(None, 0.5) == "unknown"


def test_agreement_unavailable_for_flat_or_missing_gnn():
    shap_res = {"node_ids": ["a", "b"], "node_scores": [1.0, 0.1]}
    assert not explainer_agreement(shap_res, None)["available"]
    flat = {"available": True, "flat_mask": True}
    assert not explainer_agreement(shap_res, flat)["available"]


def test_agreement_detects_match_and_mismatch():
    shap_res = {"node_ids": ["a", "b", "c", "d"], "node_scores": [0.9, 0.1, 0.2, 0.05]}
    match = {
        "available": True,
        "flat_mask": False,
        "important_nodes": [{"id": "a"}],
        "node_mask": [0.9, 0.1, 0.2, 0.05],
    }
    miss = {
        "available": True,
        "flat_mask": False,
        "important_nodes": [{"id": "d"}],
        "node_mask": [0.05, 0.2, 0.1, 0.9],
    }
    assert explainer_agreement(shap_res, match)["agree"]
    assert not explainer_agreement(shap_res, miss)["agree"]


# --- SHAP -------------------------------------------------------------------


def test_shap_is_reproducible_and_signed(model, graph):
    explainer = SHAPExplainer(model, device="cpu", seed=123)
    a = explainer.explain(graph)
    b = explainer.explain(graph)
    assert a["node_scores"] == pytest.approx(b["node_scores"], rel=1e-5, abs=1e-6)
    assert len(a["node_signed_scores"]) == len(a["node_scores"]) == 6
    # Magnitude can never be smaller than the magnitude of the signed total.
    for mag, signed in zip(a["node_scores"], a["node_signed_scores"]):
        assert mag + 1e-6 >= abs(signed)
    assert 0.0 <= a["faithfulness_random"] <= 1.0
    assert a["faithfulness_margin"] == pytest.approx(
        a["faithfulness"] - a["faithfulness_random"]
    )
    assert a["baseline"] == "self_jitter"


def test_shap_uses_benign_baseline_when_given(model, graph):
    baseline = torch.zeros(768)
    explainer = SHAPExplainer(model, device="cpu", seed=0, benign_baseline=baseline)
    result = explainer.explain(graph)
    assert result["baseline"] == "benign_mean"
    assert len(result["node_scores"]) == 6


# --- GNNExplainer -----------------------------------------------------------


def test_gnn_explainer_is_reproducible(model, graph):
    ex = GNNExplainerWrapper(model, epochs=20, device="cpu", seed=7)
    a = ex.explain(graph)
    b = ex.explain(graph)
    if not a["available"]:
        pytest.skip("PyG explain API unavailable")
    assert a["node_mask"] == pytest.approx(b["node_mask"], rel=1e-5, abs=1e-6)
    assert [n["id"] for n in a["important_nodes"]] == [n["id"] for n in b["important_nodes"]]


def test_gnn_flat_mask_reports_no_subgraph(model, graph):
    ex = GNNExplainerWrapper(model, epochs=5, device="cpu", seed=0, min_mask_spread=10.0)
    result = ex.explain(graph)
    if not result["available"]:
        pytest.skip("PyG explain API unavailable")
    assert result["flat_mask"] is True
    assert result["important_nodes"] == []
    assert result["important_edges"] == []
    assert "flat" in result["note"].lower()


# --- report wording ---------------------------------------------------------


def _shap(signed):
    ids = [f"n{i}" for i in range(len(signed))]
    return {
        "node_ids": ids,
        "node_scores": [abs(s) for s in signed],
        "node_signed_scores": signed,
        "faithfulness": 0.6,
        "faithfulness_margin": 0.4,
        "method": "shap_gradient",
        "available": True,
        "note": "",
    }


def test_report_legacy_template_without_decision(graph):
    out = ReportGenerator().generate(graph, _shap([0.5, 0.1, -0.2, 0.0, 0.0, 0.0]), None, None, None)
    assert out["explanation"].startswith("This content was flagged under")
    assert out["structured"]["decision"]["band"] == "unknown"


def test_report_safe_verdict_is_not_described_as_flagged(graph):
    out = ReportGenerator().generate(
        graph, _shap([0.5, 0.1, -0.2, 0.0, 0.0, 0.0]), None, None, None,
        harm_score=0.12, verdict="SAFE", threshold=0.5,
    )
    assert "not flagged" in out["explanation"]
    assert "was flagged" not in out["explanation"]
    assert out["structured"]["decision"]["needs_human_review"] is False


def test_report_borderline_requests_human_review(graph):
    out = ReportGenerator().generate(
        graph, _shap([0.5, 0.1, -0.2, 0.0, 0.0, 0.0]), None, None, None,
        harm_score=0.53, verdict="HARMFUL", threshold=0.5,
    )
    assert "narrowly" in out["explanation"]
    assert out["structured"]["decision"]["band"] == "borderline_harmful"
    assert out["structured"]["decision"]["needs_human_review"] is True


def test_report_lists_mitigating_elements(graph):
    out = ReportGenerator().generate(
        graph, _shap([0.5, 0.1, -0.4, -0.1, 0.0, 0.0]), None, None, None,
        harm_score=0.9, verdict="HARMFUL", threshold=0.5,
    )
    ids = [m["id"] for m in out["structured"]["mitigating_elements"]]
    assert ids[0] == "n2"
    assert "pointing toward safe" in out["explanation"]


def test_report_hedges_when_faithfulness_is_not_better_than_random(graph):
    weak = _shap([0.5, 0.1, -0.2, 0.0, 0.0, 0.0])
    weak["faithfulness"] = 0.15
    weak["faithfulness_margin"] = 0.0
    out = ReportGenerator().generate(
        graph, weak, None, None, None, harm_score=0.9, verdict="HARMFUL", threshold=0.5
    )
    assert "weakly supports" in out["explanation"]


def test_report_flags_review_when_explainers_disagree(graph):
    shap_res = _shap([0.9, 0.1, 0.2, 0.05, 0.0, 0.0])
    gnn_res = {
        "available": True,
        "flat_mask": False,
        "important_nodes": [{"id": "n5"}],
        "important_edges": [],
        "node_mask": [0.0, 0.1, 0.1, 0.2, 0.3, 0.9],
        "backend": "explain_api",
    }
    out = ReportGenerator().generate(
        graph, shap_res, gnn_res, None, None, harm_score=0.9, verdict="HARMFUL", threshold=0.5
    )
    assert out["structured"]["agreement"]["agree"] is False
    assert out["structured"]["decision"]["needs_human_review"] is True
    assert "low confidence" in out["explanation"]
