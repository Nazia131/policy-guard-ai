"""Rule-based tagging of mined harm patterns to policy categories.

:class:`SubgraphMiner` discovers *structural* harm patterns but does not know
what policy they violate.  :class:`PatternTagger` closes that gap: it inspects
the node types/labels and edge types of a mined subgraph and maps the
composition to a harm-category id, then attaches the authoritative policy
clause for that category from ``policy/policy_config.json``.

Design constraints (per the CHEM interface contract)
----------------------------------------------------
* The **category ids** a rule can emit and every **policy clause** come from
  the policy configuration via :mod:`utils.config` — never invented here.
* The **lexical keyword lists** below (``identity``, ``crowd``, ``weapon``,
  ``child``, ``contact``, ``secrecy`` …) are *detection heuristics* only:
  surface-form matchers that decide whether a node "looks like" an identity
  reference, a weapon, a child, etc.  They carry no policy meaning on their
  own; they merely route a composition to a category that must already exist
  in the configuration.

The rules encode compositional intuitions, e.g. an identity-referencing text
node co-occurring with a crowd/mob visual node is hate speech; a weapon visual
plus a person plus a ``target_pointing`` edge is a weapons threat; a child node
plus a secret-contact text node is a child-safety violation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.config import get_harm_categories, get_policy_clause

# --------------------------------------------------------------------------- #
# Lexical detection heuristics (surface-form matchers only, NOT policy).
# These decide what a node "looks like"; the emitted category id is validated
# against the policy configuration before use.
# --------------------------------------------------------------------------- #
_KEYWORDS: Dict[str, tuple] = {
    "identity": (
        "identity", "race", "racial", "ethnic", "religion", "religious",
        "gender", "sexual orientation", "nationality", "caste", "disability",
        "slur", "protected", "hate symbol", "swastika", "burning cross",
    ),
    "crowd": (
        "crowd", "mob", "group", "gang", "march", "marching", "riot",
        "protest", "masked fighters", "torches",
    ),
    "weapon": (
        "weapon", "gun", "firearm", "pistol", "rifle", "knife", "blade",
        "sword", "bomb", "explosive", "brandish",
    ),
    "person": (
        "person", "people", "man", "woman", "human", "individual", "victim",
        "someone", "child", "kid", "minor",
    ),
    "child": (
        "child", "children", "kid", "minor", "infant", "toddler", "youth",
        "underage", "playground",
    ),
    "contact": (
        "contact", "message", "dm", "meet", "meeting", "address", "location",
        "phone", "approach", "lure", "groom", "grooming",
    ),
    "secrecy": (
        "secret", "secretly", "hidden", "private", "don't tell", "keep quiet",
        "between us", "hide",
    ),
    "violence": (
        "violence", "violent", "attack", "assault", "punch", "punching",
        "kick", "kicking", "slap", "beating", "fight", "fighting", "shoot",
        "shooting", "stab", "kill", "blood", "injury", "gore",
    ),
    "harmful_text": (
        "toxic", "insult", "threat", "harass", "bully", "degrade", "mock",
        "humiliate", "abuse",
    ),
}

# --------------------------------------------------------------------------- #
# Candidate category ids the rules may emit.  Each is validated against the
# policy configuration at tag time so nothing outside the policy can leak in.
# --------------------------------------------------------------------------- #
_CAT_HATE = "hate_speech"
_CAT_WEAPONS = "weapons_threat"
_CAT_CHILD = "child_safety"
_CAT_VIOLENT = "violent_threat"
_CAT_HARASSMENT = "harassment_bullying"

# Edge-type name that expresses "this harmful element is aimed at that target".
_EDGE_TARGET_POINTING = "target_pointing"

# Node type groups.
_VISUAL_TYPES = frozenset({"image", "video"})
_TEXT_TYPES = frozenset({"text"})
_ACTION_TYPES = frozenset({"action"})


class PatternTagger:
    """Assign policy categories and clauses to mined subgraph patterns.

    Parameters
    ----------
    config_path : str, optional
        Optional override path for the policy configuration; forwarded to the
        :mod:`utils.config` accessors.
    verbose : bool, optional
        When ``True`` (default) print ``[CHEM][pattern_tagger]`` progress logs.
    """

    def __init__(self, config_path: Optional[str] = None, verbose: bool = True) -> None:
        """Initialise the tagger and cache the set of valid harm-category ids."""
        self._config_path = config_path
        self.verbose = bool(verbose)
        # The authoritative set of harm-category ids; a rule outcome is only
        # accepted if it is present here.
        self._valid_categories = set(get_harm_categories(config_path).keys())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def tag(self, pattern: Dict[str, Any]) -> str:
        """Return the harm-category id best describing a single pattern.

        Parameters
        ----------
        pattern : dict
            A mined pattern whose ``"graph"`` value is a NetworkX node-link
            JSON dict (as produced by :class:`SubgraphMiner`).

        Returns
        -------
        str
            A harm-category id drawn from the policy configuration.  If the
            preferred id for a fired rule is not present in the configuration a
            configuration-valid fallback is returned instead.
        """
        nodes, edges = _extract_nodes_edges(pattern)

        # Bucket the nodes by the heuristics that matter to the rules.
        identity_text = [n for n in nodes if _is_text(n) and _has(n, "identity")]
        crowd_visual = [n for n in nodes if _is_visual(n) and _has(n, "crowd")]
        weapon_visual = [n for n in nodes if _is_visual(n) and _has(n, "weapon")]
        person_nodes = [n for n in nodes if _has(n, "person")]
        child_nodes = [n for n in nodes if _has(n, "child")]
        contact_secret_text = [
            n
            for n in nodes
            if _is_text(n) and (_has(n, "contact") or _has(n, "secrecy"))
        ]
        violence_action = [
            n for n in nodes if _is_action(n) and _has(n, "violence")
        ]
        harmful_text = [
            n
            for n in nodes
            if _is_text(n)
            and (_node_is_harmful(n) or _has(n, "harmful_text") or _has(n, "violence"))
        ]

        has_target_pointing = any(
            _edge_is(edge, _EDGE_TARGET_POINTING) for edge in edges
        )

        # ---- Rule cascade (most specific / most severe first) ------------ #
        # 1. Child + secret contact -> child safety.
        if child_nodes and contact_secret_text:
            return self._resolve(_CAT_CHILD, nodes)

        # 2. Weapon visual + person + target_pointing -> weapons threat.
        if weapon_visual and person_nodes and has_target_pointing:
            return self._resolve(_CAT_WEAPONS, nodes)

        # 3. Identity text + crowd/mob visual -> hate speech.
        if identity_text and crowd_visual:
            return self._resolve(_CAT_HATE, nodes)

        # 4. Violent action aimed at a target -> violent threat.
        if violence_action and has_target_pointing:
            return self._resolve(_CAT_VIOLENT, nodes)

        # 5. A weapon visual on its own aimed at something -> weapons threat.
        if weapon_visual and has_target_pointing:
            return self._resolve(_CAT_WEAPONS, nodes)

        # 6. Harmful text dominates -> harassment / bullying (default harm).
        if harmful_text:
            return self._resolve(_CAT_HARASSMENT, nodes)

        # 7. Fallback: reuse any policy_category already carried by a node,
        #    otherwise default to harassment/bullying.
        carried = _first_carried_category(nodes, self._valid_categories)
        if carried:
            return carried
        return self._resolve(_CAT_HARASSMENT, nodes)

    def tag_all(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tag every pattern in place with category id and policy clause.

        Each pattern gains two keys:

        * ``policy_category`` — the harm-category id from :meth:`tag`.
        * ``policy_clause``   — the clause dict from
          :func:`utils.config.get_policy_clause`.

        Parameters
        ----------
        patterns : list of dict
            Mined patterns to annotate.

        Returns
        -------
        list of dict
            The same list, with annotations attached (also returned for
            convenience / chaining).
        """
        patterns = list(patterns or [])
        if self.verbose:
            print(f"[CHEM][pattern_tagger] Tagging {len(patterns)} patterns.")

        counts: Dict[str, int] = {}
        for pattern in patterns:
            category = self.tag(pattern)
            pattern["policy_category"] = category
            pattern["policy_clause"] = get_policy_clause(category, self._config_path)
            counts[category] = counts.get(category, 0) + 1

        if self.verbose:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"[CHEM][pattern_tagger] Done. Category counts: {summary or 'none'}")
        return patterns

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve(self, preferred: str, nodes: List[Dict[str, Any]]) -> str:
        """Return ``preferred`` if it is a valid policy category, else a fallback.

        Parameters
        ----------
        preferred : str
            The category id the fired rule would like to emit.
        nodes : list of dict
            The pattern's node dicts, consulted for a carried category fallback.

        Returns
        -------
        str
            A harm-category id guaranteed to exist in the policy configuration
            (unless the configuration defines no harm categories at all, in
            which case ``preferred`` is returned unchanged).
        """
        if preferred in self._valid_categories:
            return preferred
        carried = _first_carried_category(nodes, self._valid_categories)
        if carried:
            return carried
        if self._valid_categories:
            # Deterministic fallback to a stable, existing category.
            return sorted(self._valid_categories)[0]
        return preferred


# --------------------------------------------------------------------------- #
# Node/edge inspection helpers
# --------------------------------------------------------------------------- #
def _extract_nodes_edges(pattern: Dict[str, Any]):
    """Return ``(nodes, edges)`` lists from a pattern's node-link JSON.

    Parameters
    ----------
    pattern : dict
        A mined pattern dict containing a ``"graph"`` node-link JSON value.

    Returns
    -------
    tuple(list, list)
        The node dicts and edge dicts.  Empty lists are returned when the
        pattern lacks a usable graph payload.
    """
    graph = pattern.get("graph") or {}
    nodes = list(graph.get("nodes", []) or [])
    # node-link JSON stores edges under "links" (or "edges" on newer NetworkX).
    edges = list(graph.get("links", graph.get("edges", [])) or [])
    return nodes, edges


def _node_text(node: Dict[str, Any]) -> str:
    """Return a lowercase blob of a node's descriptive fields for matching.

    Parameters
    ----------
    node : dict
        A node dict from node-link JSON.

    Returns
    -------
    str
        Concatenated lowercase ``label``, ``full_label``, ``type``,
        ``modality`` and ``policy_category`` fields.
    """
    parts = [
        str(node.get(key, ""))
        for key in ("label", "full_label", "type", "modality", "policy_category", "text")
    ]
    return " ".join(parts).lower()


def _has(node: Dict[str, Any], bucket: str) -> bool:
    """Return whether a node matches the keyword ``bucket`` heuristic.

    Parameters
    ----------
    node : dict
        A node dict.
    bucket : str
        A key of :data:`_KEYWORDS`.

    Returns
    -------
    bool
        ``True`` if any keyword in the bucket appears in the node's text blob.
    """
    blob = _node_text(node)
    return any(kw in blob for kw in _KEYWORDS.get(bucket, ()))


def _node_type(node: Dict[str, Any]) -> str:
    """Return a node's ``type`` (fallback ``modality``) lowercased.

    Parameters
    ----------
    node : dict
        A node dict.

    Returns
    -------
    str
        The node type string, or ``""`` when unknown.
    """
    return str(node.get("type", node.get("modality", ""))).lower()


def _is_text(node: Dict[str, Any]) -> bool:
    """Return ``True`` if the node is a text node."""
    return _node_type(node) in _TEXT_TYPES


def _is_visual(node: Dict[str, Any]) -> bool:
    """Return ``True`` if the node is an image or video (visual) node."""
    return _node_type(node) in _VISUAL_TYPES


def _is_action(node: Dict[str, Any]) -> bool:
    """Return ``True`` if the node is an action node."""
    return _node_type(node) in _ACTION_TYPES


def _node_is_harmful(node: Dict[str, Any]) -> bool:
    """Return whether a node is flagged harmful via ``harmful_signal``.

    Parameters
    ----------
    node : dict
        A node dict.

    Returns
    -------
    bool
        The truthiness of the node's ``harmful_signal`` attribute.
    """
    return bool(node.get("harmful_signal", False))


def _edge_is(edge: Dict[str, Any], name: str) -> bool:
    """Return whether an edge has the given canonical edge-type name.

    Accepts either a named ``edge_type`` string or an integer id (resolved to
    its canonical name via :mod:`graph.edge_types`, imported lazily).

    Parameters
    ----------
    edge : dict
        An edge dict from node-link JSON.
    name : str
        The canonical edge-type name to test for (e.g. ``"target_pointing"``).

    Returns
    -------
    bool
        ``True`` if the edge's type matches ``name``.
    """
    value = edge.get("edge_type", edge.get("type"))
    if value is None:
        return False
    if isinstance(value, str):
        return value == name
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        try:
            from graph.edge_types import edge_type_name

            return edge_type_name(int(value)) == name
        except Exception:
            return False
    return False


def _first_carried_category(
    nodes: List[Dict[str, Any]], valid: set
) -> Optional[str]:
    """Return the first valid ``policy_category`` carried by any node.

    Parameters
    ----------
    nodes : list of dict
        The pattern's node dicts.
    valid : set
        The set of valid harm-category ids from the policy configuration.

    Returns
    -------
    str or None
        The first node-carried category id that is a recognised harm category,
        or ``None`` if none qualify.
    """
    for node in nodes:
        cat = node.get("policy_category")
        if cat and cat in valid:
            return str(cat)
    return None
