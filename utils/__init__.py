"""Shared utility package for CHEM.

Contains cross-cutting helpers that are used by every subsystem:
    * ``config``      -- load and cache ``policy/policy_config.json``.
    * ``embeddings``  -- normalise heterogeneous encoder embeddings to a fixed
                         dimension so that all modalities share a node feature
                         space in the Content Composition Graph.

Keeping these helpers in one place guarantees that harm semantics and numeric
coefficients live only in ``policy_config.json`` and never leak into Python
source files.
"""
