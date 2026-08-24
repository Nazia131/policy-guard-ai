"""Edge-type definitions for the CHEM Content Composition Graph.

CHEM models a piece of multimodal content as a *Content Composition Graph* whose
nodes are per-modality elements (text / image / audio / video / action) and whose
edges describe how those elements relate.  There are exactly five relation types
and their canonical integer ids live in ``policy/policy_config.json`` under the
``edge_types`` key so that no edge semantics are ever hardcoded in Python.

This module exposes:

``EDGE_TYPES``
    ``{name: int}`` mapping loaded from the policy config.
``EdgeType``
    An :class:`enum.IntEnum` mirroring that mapping so callers can write
    ``EdgeType.harmful_pattern`` and get the correct integer id.
``edge_type_name(i)``
    Integer id -> canonical edge-type name.
``edge_type_id(name)``
    Canonical edge-type name -> integer id.

The five edge types (canonical ids):

===  ==================  =====================================================
id   name                meaning
===  ==================  =====================================================
0    co_occurrence       same-modality elements adjacent in extraction order
1    semantic_reference  elements of different modalities (cross-modal link)
2    temporal_sequence   time-ordered video / audio / action elements
3    target_pointing     a harmful text signal aimed at a visual subject
4    harmful_pattern     both endpoints carry a harmful signal (overrides)
===  ==================  =====================================================
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

from utils.config import load_policy_config


def _load_edge_types(path: str | None = None) -> Dict[str, int]:
    """Load the canonical ``{edge_type_name: int}`` mapping from the config.

    Parameters
    ----------
    path : str, optional
        Optional override for the policy config path.

    Returns
    -------
    dict
        Mapping of edge-type name to integer id, exactly as declared in
        ``policy/policy_config.json``.

    Raises
    ------
    KeyError
        If the ``edge_types`` block is missing from the configuration.
    """
    config = load_policy_config(path)
    edge_types = config.get("edge_types")
    if not edge_types:
        raise KeyError("'edge_types' block is missing from policy_config.json")
    # Coerce ids to plain ints so downstream tensor construction is safe.
    return {str(name): int(idx) for name, idx in edge_types.items()}


# Canonical mapping loaded once at import time.  All graph code imports this so
# there is a single source of truth for edge ids across the whole system.
EDGE_TYPES: Dict[str, int] = _load_edge_types()


# Build an IntEnum dynamically from the config-driven mapping.  Using the
# functional ``IntEnum(...)`` form keeps the enum members perfectly in sync with
# ``policy_config.json`` instead of duplicating the ids in Python.
EdgeType: "IntEnum" = IntEnum("EdgeType", EDGE_TYPES)
EdgeType.__doc__ = (
    "IntEnum of CHEM Content Composition Graph edge types. Members and their "
    "integer values are loaded from the ``edge_types`` block of "
    "``policy/policy_config.json`` (co_occurrence=0, semantic_reference=1, "
    "temporal_sequence=2, target_pointing=3, harmful_pattern=4)."
)


# Reverse lookup table (int -> name) built once for O(1) name resolution.
_ID_TO_NAME: Dict[int, str] = {idx: name for name, idx in EDGE_TYPES.items()}


def edge_type_name(i: int) -> str:
    """Return the canonical edge-type name for an integer id.

    Parameters
    ----------
    i : int
        An edge-type integer id (e.g. ``4``).

    Returns
    -------
    str
        The canonical edge-type name (e.g. ``"harmful_pattern"``).

    Raises
    ------
    KeyError
        If ``i`` is not a known edge-type id.
    """
    key = int(i)
    if key not in _ID_TO_NAME:
        raise KeyError(
            f"Unknown edge-type id {i!r}; valid ids are {sorted(_ID_TO_NAME)}"
        )
    return _ID_TO_NAME[key]


def edge_type_id(name: str) -> int:
    """Return the integer id for a canonical edge-type name.

    Parameters
    ----------
    name : str
        A canonical edge-type name (e.g. ``"co_occurrence"``).

    Returns
    -------
    int
        The matching integer id (e.g. ``0``).

    Raises
    ------
    KeyError
        If ``name`` is not a known edge-type name.
    """
    if name not in EDGE_TYPES:
        raise KeyError(
            f"Unknown edge-type name {name!r}; valid names are {sorted(EDGE_TYPES)}"
        )
    return EDGE_TYPES[name]
