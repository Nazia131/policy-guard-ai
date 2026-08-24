"""Cross-module policy consistency tests.

Validates that the inter-module contract is maintained:
- Every harm category in the config has a policy clause.
- All visual concept categories resolve to defined harm/safe categories.
- All modality fusion weights normalise properly.
- All edge types in config match EdgeType enum.
- No Python file contains banned hardcoded harm strings.
"""

import os
import re

from utils.config import (
    get_coefficient,
    get_harm_categories,
    get_policy_clause,
    get_visual_concepts,
    load_policy_config,
)
from graph.edge_types import EDGE_TYPES, edge_type_id, edge_type_name


def test_policy_clauses_cover_all_harm_categories():
    """Every harm category must have a matching policy clause (and vice versa)."""
    cfg = load_policy_config()
    assert set(cfg["harm_categories"].keys()) == set(cfg["policy_clauses"].keys())


def test_every_policy_clause_has_required_fields():
    """Each policy clause must contain clause_id, title, source, section, text."""
    for cat_id in get_harm_categories():
        clause = get_policy_clause(cat_id)
        assert clause["clause_id"] != "UNMAPPED", cat_id
        for field in ("title", "source", "section", "text"):
            assert field in clause, f"{cat_id} clause missing '{field}'"


def test_visual_concepts_reference_defined_categories():
    """All CLIP visual concept categories must reference defined harm/safe categories."""
    cfg = load_policy_config()
    valid_cats = set(cfg["harm_categories"]) | set(cfg["safe_categories"])
    for concept in get_visual_concepts()["harmful"]:
        assert concept["category"] in valid_cats, concept
    for concept in get_visual_concepts()["safe"]:
        assert concept["category"] in valid_cats, concept


def test_video_action_mapping_references_valid_categories():
    """All video action -> category mappings must target valid harm categories."""
    cfg = load_policy_config()
    harm_cats = set(cfg["harm_categories"])
    for action, cat in cfg["video_action_mapping"]["harmful_actions"].items():
        assert cat in harm_cats, f"action '{action}' maps to unknown category '{cat}'"


def test_toxic_bert_mapping_references_valid_categories():
    """All toxic-bert label -> category mappings must target valid harm categories."""
    cfg = load_policy_config()
    mapping = cfg["toxic_bert_mapping"]
    harm_cats = set(cfg["harm_categories"])
    for label, cat in mapping["labels"].items():
        assert cat in harm_cats, f"toxic-bert '{label}' maps to unknown category '{cat}'"
    for lbl in mapping["text_harm_score_labels"]:
        assert lbl in mapping["labels"], f"score label '{lbl}' not in labels"


def test_scorer_weights_sum_to_one():
    """GAT probability and pattern-similarity weights must sum to 1.0."""
    total = get_coefficient("gat_vs_pattern_weight") + get_coefficient("pattern_similarity_weight")
    assert abs(total - 1.0) < 1e-9, total


def test_audio_weights_sum_to_one():
    """Audio transcript and acoustic weights must form a convex combination."""
    total = get_coefficient("audio_transcript_weight") + get_coefficient("audio_acoustic_weight")
    assert abs(total - 1.0) < 1e-9, total


def test_edge_types_enum_matches_config():
    """EdgeType enum values must match the config edge_types mapping exactly."""
    cfg = load_policy_config()
    for name, expected_id in cfg["edge_types"].items():
        assert edge_type_id(name) == expected_id
        assert edge_type_name(expected_id) == name


def test_no_hardcoded_visual_prompts_in_python():
    """No Python source should hardcode CLIP visual-concept prompt strings.

    Policy clause ids and category ids used as *validated* rule targets (in
    ``harm_pattern_library/pattern_tagger.py``) are acceptable because they are
    checked against ``get_harm_categories()`` at runtime; raw CLIP prompt text
    and clause ids must not appear in Python source.
    """
    # CLIP prompt substrings that must only live in policy_config.json.
    forbidden_patterns = [
        r'"a person holding a weapon and threatening"',
        r'"a person pointing a gun"',
        r'"MCS-3\.2"',
        r'"YT-CSE-1"',
        r'"DSA-Art34"',
    ]

    # Files with legitimate config-adjacent references (e.g. example in docstring,
    # the config itself, or test utilities that call config accessors by name).
    excluded_suffixes = {
        os.path.normpath("policy/policy_config.json"),
        os.path.normpath("tests/test_policy_consistency.py"),
        os.path.normpath("tests/test_config_and_embeddings.py"),
    }

    violations: list = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.normpath(os.path.join(root, fname))
            if fpath in excluded_suffixes:
                continue
            content = open(fpath).read()
            for pattern in forbidden_patterns:
                if re.search(pattern, content):
                    violations.append(f"{fpath}: {pattern}")

    assert not violations, "Hardcoded harm strings found:\n" + "\n".join(violations)


def test_pattern_tagger_category_constants_are_config_validated():
    """PatternTagger's category-id constants must all exist in the policy config.

    ``pattern_tagger.py`` deliberately defines category-id constants as rule
    targets.  This test confirms every such constant is validated against
    ``get_harm_categories()`` at runtime so config changes propagate correctly.
    """
    from harm_pattern_library.pattern_tagger import (
        _CAT_CHILD,
        _CAT_HARASSMENT,
        _CAT_HATE,
        _CAT_VIOLENT,
        _CAT_WEAPONS,
    )

    valid = set(get_harm_categories().keys())
    for const in (_CAT_CHILD, _CAT_HARASSMENT, _CAT_HATE, _CAT_VIOLENT, _CAT_WEAPONS):
        assert const in valid, f"PatternTagger constant '{const}' not in policy config"
