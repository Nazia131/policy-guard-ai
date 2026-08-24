"""Central configuration loader for CHEM.

Every subsystem in CHEM reads its harm categories, visual concepts, policy
clauses and numeric coefficients from ``policy/policy_config.json``.  This module
loads that file exactly once (LRU-cached) and exposes small typed accessors so
that no module has to know the on-disk layout or hardcode any harm semantics.

Public API
----------
load_policy_config(path=None) -> dict
    Return the full parsed configuration dictionary (cached).
get_coefficient(name, path=None) -> float
    Return a single tunable numeric coefficient.
get_harm_categories(path=None) -> dict
    Return the mapping of harm-category id -> definition.
get_policy_clause(category_id, path=None) -> dict
    Return the policy clause metadata for a harm category id.
get_visual_concepts(path=None) -> dict
    Return the ``{"harmful": [...], "safe": [...]}`` CLIP concept prompts.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

# Absolute path to the bundled policy configuration.  Resolved relative to this
# file so the loader works regardless of the current working directory.
_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "policy", "policy_config.json")
)


@lru_cache(maxsize=8)
def load_policy_config(path: str | None = None) -> Dict[str, Any]:
    """Load and cache the CHEM policy configuration.

    Parameters
    ----------
    path : str, optional
        Path to the JSON config.  Defaults to ``policy/policy_config.json``
        located next to the project root.

    Returns
    -------
    dict
        The fully parsed configuration.  The result is cached, so repeated
        calls are effectively free and every subsystem shares one instance.

    Raises
    ------
    FileNotFoundError
        If the configuration file cannot be located.
    json.JSONDecodeError
        If the configuration file is not valid JSON.
    """
    resolved = path or _DEFAULT_CONFIG_PATH
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"CHEM policy config not found at: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_coefficient(name: str, path: str | None = None) -> float:
    """Return a single tunable numeric coefficient by name.

    Parameters
    ----------
    name : str
        Key under the ``coefficients`` object in the config.
    path : str, optional
        Optional override for the config path.

    Returns
    -------
    float
        The coefficient value.

    Raises
    ------
    KeyError
        If ``name`` is not present in the ``coefficients`` block.
    """
    coefficients = load_policy_config(path).get("coefficients", {})
    if name not in coefficients:
        raise KeyError(f"Coefficient '{name}' is not defined in policy_config.json")
    return float(coefficients[name])


def get_harm_categories(path: str | None = None) -> Dict[str, Any]:
    """Return the mapping of harm-category id to its definition.

    Parameters
    ----------
    path : str, optional
        Optional override for the config path.

    Returns
    -------
    dict
        ``{category_id: {"display_name": ..., "severity": ..., ...}}``.
    """
    return load_policy_config(path).get("harm_categories", {})


def get_policy_clause(category_id: str, path: str | None = None) -> Dict[str, Any]:
    """Return the policy clause metadata for a harm-category id.

    Parameters
    ----------
    category_id : str
        A harm-category id such as ``"hate_speech"``.
    path : str, optional
        Optional override for the config path.

    Returns
    -------
    dict
        Clause metadata (``clause_id``, ``title``, ``source``, ``section``,
        ``text``).  Returns an ``{"clause_id": "UNMAPPED", ...}`` placeholder if
        the category has no mapped clause so callers never raise on lookup.
    """
    clauses = load_policy_config(path).get("policy_clauses", {})
    return clauses.get(
        category_id,
        {
            "clause_id": "UNMAPPED",
            "title": "Unmapped category",
            "source": "N/A",
            "section": "N/A",
            "text": f"No policy clause is mapped to category '{category_id}'.",
        },
    )


def get_visual_concepts(path: str | None = None) -> Dict[str, Any]:
    """Return CLIP zero-shot concept prompts grouped by harmfulness.

    Parameters
    ----------
    path : str, optional
        Optional override for the config path.

    Returns
    -------
    dict
        ``{"harmful": [{"prompt": ..., "category": ...}, ...],
           "safe":    [{"prompt": ..., "category": ...}, ...]}``.
    """
    return load_policy_config(path).get("visual_concepts", {"harmful": [], "safe": []})
