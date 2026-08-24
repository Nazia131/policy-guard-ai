"""Embedding normalisation helpers for CHEM.

Different encoders emit embeddings of different dimensions (RoBERTa -> 768,
CLIP ViT-B/32 -> 512, VideoMAE -> 768, acoustic feature vectors -> small).
The Content Composition Graph, however, requires every node to live in a single
feature space so the GAT can consume a homogeneous ``x`` matrix.

This module provides one canonical operation -- :func:`to_fixed_dim` -- which
projects any vector to the configured embedding dimension (default 768) by
zero-padding shorter vectors and truncating longer ones.  Zero-padding is a
deterministic, information-preserving lift that keeps the original dimensions
intact, which is important for downstream explainability (SHAP / GNNExplainer)
where node feature indices must remain interpretable.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .config import load_policy_config


def target_dim(path: str | None = None) -> int:
    """Return the configured node feature dimension.

    Parameters
    ----------
    path : str, optional
        Optional override for the policy config path.

    Returns
    -------
    int
        The ``meta.embedding_dim`` value from ``policy_config.json`` (768).
    """
    return int(load_policy_config(path).get("meta", {}).get("embedding_dim", 768))


def to_fixed_dim(vector: Sequence[float], dim: int | None = None) -> List[float]:
    """Normalise an arbitrary numeric vector to a fixed dimension.

    Shorter vectors are zero-padded on the right; longer vectors are truncated.
    A ``None`` or empty input yields an all-zero vector so that malformed
    encoder output never crashes graph construction.

    Parameters
    ----------
    vector : sequence of float
        The raw embedding produced by an encoder.
    dim : int, optional
        Desired output dimension.  Defaults to the configured embedding
        dimension (768).

    Returns
    -------
    list of float
        A list of exactly ``dim`` floats.
    """
    out_dim = dim if dim is not None else target_dim()
    if vector is None:
        return [0.0] * out_dim
    values = [float(v) for v in vector]
    if len(values) >= out_dim:
        return values[:out_dim]
    return values + [0.0] * (out_dim - len(values))


def stack_fixed_dim(vectors: Iterable[Sequence[float]], dim: int | None = None) -> List[List[float]]:
    """Normalise a batch of vectors to a common fixed dimension.

    Parameters
    ----------
    vectors : iterable of sequence of float
        Raw embeddings from one or more encoders.
    dim : int, optional
        Desired output dimension (defaults to the configured value).

    Returns
    -------
    list of list of float
        A rectangular matrix where every row has length ``dim``.
    """
    out_dim = dim if dim is not None else target_dim()
    return [to_fixed_dim(v, out_dim) for v in vectors]
