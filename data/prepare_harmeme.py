"""Convert the MemeLens HarMeme subset into the layout CHEM trains on.

MemeLens (QCRI/MemeLens on Hugging Face) ships each dataset as parquet files
whose rows carry an embedded ``image`` plus ``text`` and a three-level ``label``
(``not-harmful`` / ``partially-harmful`` / ``very-harmful``).  CHEM's loader
(``data/preprocess/hateful_memes.py``) expects::

    <out-dir>/img/<id>.png
    <out-dir>/train.jsonl   {"id", "img", "text", "label"}   label in {0, 1}
    <out-dir>/dev.jsonl
    <out-dir>/test.jsonl

Usage::

    python -m data.prepare_harmeme --src memelens_harmeme/Harmful_Covid_en__HarMeme \\
        --out-dir data/hateful_memes

``--src`` is the folder that holds the parquet files (it is searched
recursively).  Split names are taken from the file names: ``train*``,
``val*``/``dev*`` and ``test*``.

Label collapsing (``--positive``):
    any   (default) partially-harmful + very-harmful -> 1   (~35 % positive)
    very  only very-harmful -> 1                            (~6 % positive)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

_NEGATIVE = "not-harmful"
_VERY = "very-harmful"

# Substrings of a parquet file name -> CHEM split name (first match wins).
_SPLIT_HINTS = (("train", "train"), ("validation", "dev"), ("val", "dev"),
                ("dev", "dev"), ("test", "test"))


def _log(message: str) -> None:
    """Print a namespaced progress line."""
    print(f"[CHEM][prepare_harmeme] {message}")


def _split_of(path: str) -> Optional[str]:
    """Return the CHEM split name implied by a parquet file name, or ``None``."""
    name = os.path.basename(path).lower()
    for hint, split in _SPLIT_HINTS:
        if hint in name:
            return split
    return None


def _binary_label(value: Any, positive: str) -> int:
    """Collapse a HarMeme label string to 0/1.

    Inputs:
        value: Raw label (e.g. ``"partially-harmful"``).
        positive: ``"any"`` (partial + very -> 1) or ``"very"`` (only very -> 1).
    Outputs:
        ``0`` or ``1``.
    Raises:
        ValueError: On a label that is not one of the three known values.
    """
    label = str(value).strip().lower()
    if label == _NEGATIVE:
        return 0
    if label == _VERY:
        return 1
    if label == "partially-harmful":
        return 1 if positive == "any" else 0
    raise ValueError(f"Unexpected HarMeme label: {value!r}")


def _open_image(cell: Any, src_dir: str):
    """Turn a parquet ``image`` cell into a PIL RGB image.

    Handles the Hugging Face ``{"bytes": ..., "path": ...}`` struct, raw bytes,
    and plain file-path strings (resolved against ``src_dir``).
    """
    from PIL import Image

    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        cell = cell.get("path")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    if isinstance(cell, str) and cell:
        path = cell if os.path.isabs(cell) else os.path.join(src_dir, cell)
        return Image.open(path).convert("RGB")
    raise ValueError("image cell has neither bytes nor a usable path")


def _safe_name(value: Any, fallback: str) -> str:
    """Make a filesystem-safe image stem from a row id."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or fallback


def convert(src: str, out_dir: str, positive: str = "any") -> Dict[str, Dict[str, int]]:
    """Convert every parquet split under ``src`` into the CHEM layout.

    Inputs:
        src: Folder containing the MemeLens HarMeme parquet files.
        out_dir: Destination dataset root (``img/`` and ``*.jsonl`` go here).
        positive: Label collapsing rule, ``"any"`` or ``"very"``.
    Outputs:
        ``{split: {"rows": n, "positive": k, "skipped": s}}``.
    Raises:
        FileNotFoundError: If no parquet files are found under ``src``.
    """
    import pandas as pd

    files = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {src!r}.")

    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    by_split: Dict[str, List[str]] = {}
    for path in files:
        split = _split_of(path)
        if split is None:
            _log(f"Ignoring file with unrecognised split name: {path}")
            continue
        by_split.setdefault(split, []).append(path)

    summary: Dict[str, Dict[str, int]] = {}
    for split, paths in by_split.items():
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        for column in ("image", "text", "label"):
            if column not in frame.columns:
                raise KeyError(
                    f"{split}: column {column!r} missing; found {list(frame.columns)}"
                )

        records, skipped, seen = [], 0, set()
        for idx, row in frame.iterrows():
            try:
                label = _binary_label(row["label"], positive)
                image = _open_image(row["image"], src)
            except Exception as exc:  # one bad row must not stop the conversion
                skipped += 1
                _log(f"{split} row {idx}: skipped ({exc})")
                continue

            stem = _safe_name(row.get("id", idx), str(idx))
            if (split, stem) in seen:  # ids repeated within a split
                stem = f"{stem}_{idx}"
            seen.add((split, stem))
            rel = f"img/{split}_{stem}.png"
            image.save(os.path.join(out_dir, rel))

            text = row["text"]
            records.append({
                "id": f"{split}_{stem}",
                "img": rel,
                "text": "" if text is None or text != text else str(text),
                "label": label,
            })

        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        positives = sum(r["label"] for r in records)
        summary[split] = {"rows": len(records), "positive": positives, "skipped": skipped}
        _log(f"{split}: {len(records)} rows, {positives} positive, {skipped} skipped")

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True, help="Folder with the HarMeme parquet files.")
    parser.add_argument("--out-dir", default=os.path.join("data", "hateful_memes"),
                        help="Output dataset root (default: data/hateful_memes).")
    parser.add_argument("--positive", choices=("any", "very"), default="any",
                        help="'any': partial+very -> 1 (default). 'very': only very -> 1.")
    args = parser.parse_args(argv)

    try:
        summary = convert(args.src, args.out_dir, args.positive)
    except Exception as exc:
        _log(f"FAILED: {exc}")
        return 1
    if not summary:
        _log("FAILED: no parquet file name contained train / val / dev / test.")
        return 1
    _log(f"Done. Dataset ready at {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
