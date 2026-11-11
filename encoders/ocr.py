"""OCR for text embedded in images (memes, screenshots).

Uses RapidOCR (ONNX runtime). Its detection/recognition models ship inside the
pip wheel, so no model download from GitHub or Hugging Face is needed.  The
engine loads lazily on first use and a failure never breaks the pipeline: if
OCR is unavailable, an empty string is returned and a warning is printed.

``extract_text(image) -> str`` accepts a PIL.Image and returns the recognised
lines joined by spaces, in reading order (top to bottom, left to right).
"""

from __future__ import annotations

from typing import Any, List, Optional

_engine: Any = None
_failed = False

#: Detections below this recognition confidence are discarded as noise.
MIN_CONFIDENCE = 0.5


def _get_engine():
    """Load the RapidOCR engine once; return ``None`` if it can't be loaded."""
    global _engine, _failed
    if _engine is None and not _failed:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
            print("[CHEM][ocr] RapidOCR engine ready.")
        except Exception as exc:  # noqa: BLE001
            _failed = True
            print(f"[CHEM][ocr] OCR unavailable ({exc}); install rapidocr-onnxruntime.")
    return _engine


def _reading_order(rows: List[Any]) -> List[Any]:
    """Sort detections top-to-bottom, then left-to-right within a line."""
    def key(row):
        box = row[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        height = max(ys) - min(ys) or 1.0
        return (round(min(ys) / height), min(xs))
    return sorted(rows, key=key)


def extract_text(image: Any, min_confidence: Optional[float] = None) -> str:
    """Return the text found in ``image`` ('' when none or OCR unavailable)."""
    import numpy as np

    engine = _get_engine()
    if engine is None:
        return ""
    floor = MIN_CONFIDENCE if min_confidence is None else min_confidence
    try:
        result, _ = engine(np.array(image.convert("RGB")))
    except Exception as exc:  # noqa: BLE001
        print(f"[CHEM][ocr] OCR failed: {exc}")
        return ""
    if not result:
        return ""
    rows = [r for r in result if float(r[2]) >= floor and str(r[1]).strip()]
    text = " ".join(str(r[1]).strip() for r in _reading_order(rows))
    print(f"[CHEM][ocr] extracted {len(text)} chars")
    return text
