"""Checkpoint save / load helpers and model-status logic for the CHEM GAT.

Why this module exists
----------------------
Before this module, ``model/train.py`` saved only ``state_dict`` / ``val_f1`` /
``epoch`` and ``model/harm_scorer.py`` silently fell back to random weights when
no checkpoint was found.  Nothing recorded *what* a checkpoint was trained on,
so a checkpoint trained on synthetic demo data was indistinguishable from a real
one.

Checkpoint format (version 2)
-----------------------------
A plain ``dict`` containing only basic Python types plus the state dict, so it
loads with ``torch.load(..., weights_only=True)``::

    {
      "state_dict": {...},          # model weights (unchanged key, so older
      "val_f1": float,              #  readers that use ckpt["state_dict"] still work)
      "epoch": int,
      "metadata": {                 # provenance -- see build_metadata()
          "format_version": 2, "dataset_name": str, "is_synthetic": bool, ...
      },
    }

Older checkpoints (a bare ``state_dict`` or ``{"state_dict", "val_f1", "epoch"}``)
still load; they simply have no provenance and are reported as
``"loaded_unverified"``.

Model status values
-------------------
``untrained``          no weights were loaded (random initialisation)
``loaded_unverified``  weights loaded, but no dataset provenance in the checkpoint
``synthetic_only``     weights were trained on the synthetic demo data
``trained``            weights loaded and provenance names a non-synthetic dataset

"trained" means *the provenance says the weights came from our training
pipeline on a named, non-synthetic dataset*.  It says nothing about quality;
see ``val_f1`` and held-out evaluation for that.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import torch

CHECKPOINT_FORMAT_VERSION = 2

STATUS_UNTRAINED = "untrained"
STATUS_UNVERIFIED = "loaded_unverified"
STATUS_SYNTHETIC = "synthetic_only"
STATUS_TRAINED = "trained"

_STATUS_MESSAGES: Dict[str, Optional[str]] = {
    STATUS_UNTRAINED: (
        "GAT weights were not loaded: scores come from a randomly initialised "
        "network and are not a learned moderation decision."
    ),
    STATUS_UNVERIFIED: (
        "GAT weights were loaded from a checkpoint with no dataset provenance, "
        "so what it was trained on cannot be verified."
    ),
    STATUS_SYNTHETIC: (
        "GAT weights were trained on synthetic demo data only; they do not "
        "represent a trained moderation model."
    ),
    STATUS_TRAINED: None,
}


def status_message(status: str) -> Optional[str]:
    """Return the human-readable warning for a model status (None if no warning).

    Parameters
    ----------
    status : str
        One of the ``STATUS_*`` constants.

    Returns
    -------
    str or None
        A warning sentence, or ``None`` for ``"trained"``.
    """
    return _STATUS_MESSAGES.get(status, "Unknown model status.")


def derive_model_status(weights_loaded: bool, metadata: Optional[Dict[str, Any]]) -> str:
    """Decide the model status from whether weights loaded and what the metadata says.

    Parameters
    ----------
    weights_loaded : bool
        Whether a checkpoint was successfully loaded into the model.
    metadata : dict or None
        Checkpoint metadata (may be empty for legacy checkpoints).

    Returns
    -------
    str
        One of the ``STATUS_*`` constants.
    """
    if not weights_loaded:
        return STATUS_UNTRAINED
    metadata = metadata or {}
    if metadata.get("is_synthetic") is True:
        return STATUS_SYNTHETIC
    if metadata.get("dataset_name"):
        return STATUS_TRAINED
    return STATUS_UNVERIFIED


def describe_architecture(model: torch.nn.Module) -> Dict[str, Any]:
    """Return the shape-defining hyper-parameters of a CHEM GAT instance.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The model to describe.

    Returns
    -------
    dict
        ``{"in_channels", "hidden_channels", "heads", "num_classes", "dropout"}``.
    """
    return {
        "in_channels": int(getattr(model, "in_channels")),
        "hidden_channels": int(getattr(model, "hidden_channels")),
        "heads": int(getattr(model, "heads")),
        "num_classes": int(getattr(model, "num_classes")),
        "dropout": float(getattr(model, "dropout")),
    }


def check_architecture(metadata: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> None:
    """Raise a clear error if a checkpoint's recorded architecture differs from ours.

    Only runs when the checkpoint recorded an architecture (legacy checkpoints
    did not; ``load_state_dict`` still catches shape mismatches for those).

    Parameters
    ----------
    metadata : dict or None
        Checkpoint metadata.
    expected : dict
        Architecture of the model that will receive the weights.

    Raises
    ------
    ValueError
        If any recorded shape-defining value differs from ``expected``.
    """
    recorded = (metadata or {}).get("architecture")
    if not isinstance(recorded, dict):
        return
    for key in ("in_channels", "hidden_channels", "heads", "num_classes"):
        if key in recorded and key in expected and recorded[key] != expected[key]:
            raise ValueError(
                f"Checkpoint architecture mismatch on '{key}': checkpoint has "
                f"{recorded[key]}, current model has {expected[key]}."
            )


def _git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if it cannot be determined."""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and commit else None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def build_metadata(
    *,
    dataset_name: str,
    data_dir: Optional[str],
    label_mapping: str,
    is_synthetic: bool,
    split_scheme: str,
    split_seed: int,
    split_counts: Dict[str, int],
    hyperparameters: Dict[str, Any],
    architecture: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the provenance dict stored inside a checkpoint.

    Only basic Python types are used so the checkpoint loads with
    ``torch.load(weights_only=True)``.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset actually used (e.g. ``"hateful_memes"``, ``"harmeme_v1"``).
    data_dir : str or None
        Dataset root; only its folder name is stored (no absolute local path).
    label_mapping : str
        Plain-language description of how dataset labels map to SAFE / HARMFUL.
    is_synthetic : bool
        True if the data came from the synthetic demo generator.
    split_scheme : str
        How train/val/test were produced.
    split_seed : int
        Seed used for the split shuffle.
    split_counts : dict
        ``{"train": n, "val": n, "test": n}``.
    hyperparameters : dict
        Optimiser / loop settings.
    architecture : dict
        Output of :func:`describe_architecture`.

    Returns
    -------
    dict
        The metadata dict.
    """
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset_name": str(dataset_name),
        "dataset_dir_name": os.path.basename(os.path.abspath(data_dir)) if data_dir else None,
        "is_synthetic": bool(is_synthetic),
        "label_mapping": str(label_mapping),
        "split_scheme": str(split_scheme),
        "split_seed": int(split_seed),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "hyperparameters": dict(hyperparameters),
        "architecture": dict(architecture),
        "torch_version": str(torch.__version__),
        "git_commit": _git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def save_checkpoint(
    path: str,
    state_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    val_f1: float,
    epoch: int,
) -> None:
    """Save a version-2 checkpoint (weights + provenance).

    Parameters
    ----------
    path : str
        Destination file.
    state_dict : dict
        Model weights.
    metadata : dict
        Output of :func:`build_metadata`.
    val_f1 : float
        Validation F1 at the saved epoch.
    epoch : int
        Epoch of the saved weights.
    """
    torch.save(
        {
            "state_dict": state_dict,
            "val_f1": float(val_f1),
            "epoch": int(epoch),
            "metadata": dict(metadata),
        },
        path,
    )


def read_checkpoint(path: str, map_location: Any = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint of any supported format.

    Uses ``weights_only=True`` so loading a checkpoint cannot execute arbitrary
    pickled code.

    Parameters
    ----------
    path : str
        Checkpoint file.
    map_location : optional
        Forwarded to ``torch.load``.

    Returns
    -------
    tuple(dict, dict)
        ``(state_dict, metadata)``.  ``metadata`` is empty for a bare state dict;
        for the older ``{"state_dict", "val_f1", "epoch"}`` format it contains
        just ``val_f1`` / ``epoch`` (no provenance).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        metadata = checkpoint.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        for key in ("val_f1", "epoch"):
            if key in checkpoint:
                metadata.setdefault(key, checkpoint[key])
        return checkpoint["state_dict"], metadata
    return checkpoint, {}
