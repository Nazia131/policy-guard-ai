"""Regression tests for pattern-library parsing and tagging."""

import pytest

from harm_pattern_library.pattern_tagger import PatternTagger


def _links_graph():
    """A node-link graph using the ``links`` key the miner writes."""
    return {
        "directed": True, "multigraph": False, "graph": {},
        "nodes": [
            {"id": 0, "type": "text", "harmful_signal": True},
            {"id": 1, "type": "image", "harmful_signal": False},
        ],
        "links": [{"source": 0, "target": 1, "edge_type": 3}],
    }


def test_pattern_graph_with_links_key_is_parsed():
    pytest.importorskip("torch")
    from model.harm_scorer import HarmScorer

    graph = HarmScorer._pattern_to_nx({"graph": _links_graph()})
    assert graph is not None
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 1


def test_tagger_keeps_category_carried_by_harmful_text_node():
    tagger = PatternTagger(verbose=False)
    pattern = {
        "graph": {
            "nodes": [
                {"id": 0, "type": "text", "label": "hate_speech",
                 "harmful_signal": True, "policy_category": "hate_speech"},
            ],
            "links": [],
        }
    }
    assert tagger.tag(pattern) == "hate_speech"
