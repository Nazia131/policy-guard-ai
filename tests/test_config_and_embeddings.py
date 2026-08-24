"""Tests for the CHEM configuration contract and embedding normalisation.

Run with: ``pytest tests/test_config_and_embeddings.py``
These tests require no heavy ML dependencies.
"""

from utils.config import (
    get_coefficient,
    get_harm_categories,
    get_policy_clause,
    get_visual_concepts,
    load_policy_config,
)
from utils.embeddings import stack_fixed_dim, target_dim, to_fixed_dim


def test_config_has_all_required_harm_categories():
    """Every harm category required by the spec must be present in the config."""
    required = {
        "hate_speech",
        "violent_threat",
        "child_safety",
        "weapons_threat",
        "harassment_bullying",
        "graphic_violence",
        "dangerous_misinformation",
        "extremist_content",
        "sexual_exploitation",
        "coordinated_harm",
    }
    assert required.issubset(set(get_harm_categories().keys()))


def test_every_harm_category_maps_to_a_policy_clause():
    """Each harm category must resolve to a non-placeholder policy clause."""
    for category_id in get_harm_categories():
        clause = get_policy_clause(category_id)
        assert clause["clause_id"] != "UNMAPPED", category_id
        assert clause["source"] in {
            "Meta Community Standards",
            "YouTube Community Guidelines",
            "EU Digital Services Act",
        }


def test_visual_concepts_are_action_based_and_labelled():
    """Harmful CLIP concepts must be action-based prompts tagged with a category."""
    concepts = get_visual_concepts()
    assert concepts["harmful"] and concepts["safe"]
    for concept in concepts["harmful"]:
        assert "prompt" in concept and "category" in concept
        # Action-based prompts are multi-word phrases, not bare object names.
        assert len(concept["prompt"].split()) >= 4


def test_required_coefficients_present():
    """All tunable coefficients referenced by modules must exist."""
    for name in [
        "image_harmful_min_confidence",
        "final_harm_threshold",
        "cross_modal_bonus",
        "cross_modal_min_image_score",
        "cross_modal_min_audio_score",
        "audio_transcript_weight",
        "audio_acoustic_weight",
        "gat_vs_pattern_weight",
        "pattern_similarity_weight",
        "co_occurrence_window",
    ]:
        assert isinstance(get_coefficient(name), float)


def test_audio_weights_sum_to_one():
    """Audio transcript/acoustic weights should form a convex combination."""
    total = get_coefficient("audio_transcript_weight") + get_coefficient("audio_acoustic_weight")
    assert abs(total - 1.0) < 1e-9


def test_embedding_dim_is_768():
    """The configured node feature dimension must be 768 (RoBERTa size)."""
    assert target_dim() == 768
    assert load_policy_config()["meta"]["embedding_dim"] == 768


def test_to_fixed_dim_pads_and_truncates():
    """Short vectors are zero-padded; long vectors are truncated; all -> 768."""
    assert to_fixed_dim([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0] + [0.0] * 765
    assert len(to_fixed_dim(list(range(2000)))) == 768
    assert to_fixed_dim(None) == [0.0] * 768


def test_stack_fixed_dim_is_rectangular():
    """Batched normalisation must yield a rectangular 768-wide matrix."""
    matrix = stack_fixed_dim([[1.0], [1.0, 2.0, 3.0], list(range(900))])
    assert all(len(row) == 768 for row in matrix)
    assert len(matrix) == 3
