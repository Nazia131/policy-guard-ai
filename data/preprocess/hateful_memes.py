#!/usr/bin/env python3
"""Loader utilities for the Facebook Hateful Memes dataset.

The Hateful Memes dataset ships one JSON-lines file per split (``train.jsonl``,
``dev.jsonl``, ``test.jsonl``) plus an ``img/`` directory of meme images. Each
line of a split file is a JSON object with the fields:

    id    : int/str  -- unique sample identifier
    img   : str      -- image path RELATIVE to the dataset root (e.g. "img/42953.png")
    text  : str      -- the overlaid meme caption
    label : int      -- 0 (not hateful) or 1 (hateful); absent for blind test sets

This module exposes:

* :func:`load_hateful_memes` — a generator that yields normalised sample dicts
  with an ABSOLUTE image path.
* :class:`HatefulMemesDataset` — a light, index-able wrapper (``__len__`` /
  ``__getitem__``) suitable for feeding a PyTorch/PyG training loop.

The module imports cleanly with only the standard library; missing dataset
files raise a single, informative :class:`FileNotFoundError` rather than an
opaque crash.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator


# Splits and the alternative filenames the community distribution sometimes uses.
_SPLIT_FILENAMES: dict[str, list[str]] = {
    "train": ["train.jsonl"],
    "dev": ["dev.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl"],
    "test": ["test.jsonl", "test_seen.jsonl", "test_unseen.jsonl"],
}


def _log(message: str) -> None:
    """Print a namespaced progress line.

    Inputs:
        message: Human-readable progress text.
    Outputs:
        None. Writes ``[CHEM][hateful_memes] ...`` to stdout.
    """
    print(f"[CHEM][hateful_memes] {message}")


def _resolve_split_file(data_dir: str, split: str) -> str:
    """Locate the ``.jsonl`` file for a given split inside ``data_dir``.

    Inputs:
        data_dir: Dataset root directory (contains the *.jsonl files and img/).
        split: One of ``"train"``, ``"dev"`` or ``"test"``.
    Outputs:
        Absolute path to the resolved split file.
    Raises:
        ValueError: If ``split`` is not a recognised split name.
        FileNotFoundError: If ``data_dir`` or no candidate split file exists,
            with guidance on how to obtain the dataset.
    """
    if split not in _SPLIT_FILENAMES:
        raise ValueError(
            f"Unknown split {split!r}. Expected one of {list(_SPLIT_FILENAMES)}."
        )

    abs_root = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(
            f"Hateful Memes data directory not found: {abs_root}\n"
            "Obtain the dataset first (see data/download_datasets.py: "
            "`python3 data/download_datasets.py --dataset hateful_memes`)."
        )

    for candidate in _SPLIT_FILENAMES[split]:
        path = os.path.join(abs_root, candidate)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"No split file for {split!r} found in {abs_root}. "
        f"Looked for: {_SPLIT_FILENAMES[split]}. "
        "Ensure the dataset was extracted with its train/dev/test .jsonl files."
    )


def load_hateful_memes(data_dir: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """Yield normalised Hateful Memes samples for a split.

    Reads the split's ``.jsonl`` file line by line and yields one dict per
    sample. The image path is resolved to an ABSOLUTE path so downstream
    encoders can open it regardless of the current working directory.

    Inputs:
        data_dir: Dataset root directory containing the ``.jsonl`` files and
            the ``img/`` folder.
        split: Which split to read (``"train"``, ``"dev"``, or ``"test"``).
    Yields:
        Dicts of the form::

            {
              "id":    str,   # sample id as a string
              "img":   str,   # ABSOLUTE path to the meme image
              "text":  str,   # meme caption
              "label": int,   # 0 or 1; -1 if the split has no labels (blind test)
            }

    Raises:
        FileNotFoundError: If the data directory or split file is missing.
    """
    split_file = _resolve_split_file(data_dir, split)
    abs_root = os.path.dirname(split_file)
    _log(f"Loading split '{split}' from {split_file}")

    count = 0
    skipped = 0
    with open(split_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                _log(f"Skipping malformed JSON on line {line_no}: {exc}")
                continue

            rel_img = record.get("img", "")
            abs_img = os.path.join(abs_root, rel_img) if rel_img else ""

            yield {
                "id": str(record.get("id", line_no)),
                "img": abs_img,
                "text": record.get("text", ""),
                # Blind test splits omit 'label'; use -1 as an explicit sentinel.
                "label": int(record["label"]) if "label" in record else -1,
            }
            count += 1

    _log(f"Yielded {count} samples from '{split}' ({skipped} malformed lines skipped).")


class HatefulMemesDataset:
    """Index-able, in-memory wrapper around a Hateful Memes split.

    Eagerly materialises the split into a list of normalised sample dicts (the
    ``.jsonl`` files are small metadata, not the images themselves) and exposes
    a minimal sequence protocol so the dataset can be iterated, ``len()``-ed and
    indexed like a PyTorch ``Dataset``.

    Attributes:
        data_dir: Dataset root directory.
        split: The split this instance represents.
        samples: List of normalised sample dicts (see :func:`load_hateful_memes`).
    """

    def __init__(self, data_dir: str, split: str = "train") -> None:
        """Load and cache a split.

        Inputs:
            data_dir: Dataset root directory.
            split: Split name (``"train"``, ``"dev"``, ``"test"``).
        Raises:
            FileNotFoundError: If the data directory or split file is missing.
        """
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.split = split
        self.samples: list[dict[str, Any]] = list(
            load_hateful_memes(self.data_dir, split)
        )
        _log(f"HatefulMemesDataset ready: {len(self.samples)} samples ('{split}').")

    def __len__(self) -> int:
        """Return the number of samples in this split.

        Outputs:
            Integer sample count.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the sample at ``index``.

        Inputs:
            index: Zero-based position into the split.
        Outputs:
            The normalised sample dict at ``index``.
        Raises:
            IndexError: If ``index`` is out of range.
        """
        return self.samples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in order.

        Outputs:
            An iterator over the normalised sample dicts.
        """
        return iter(self.samples)


if __name__ == "__main__":
    # Tiny smoke-test CLI: print a couple of samples if a dir is supplied.
    import argparse

    _parser = argparse.ArgumentParser(description="Inspect a Hateful Memes split.")
    _parser.add_argument("data_dir", help="Dataset root directory.")
    _parser.add_argument("--split", default="train")
    _parser.add_argument("--n", type=int, default=3, help="How many samples to show.")
    _args = _parser.parse_args()

    _ds = HatefulMemesDataset(_args.data_dir, _args.split)
    for _i in range(min(_args.n, len(_ds))):
        print(_ds[_i])
