"""Tests for the MemeLens HarMeme adapter."""

import io
import json
import os

import pytest

from data.prepare_harmeme import _binary_label, convert


@pytest.mark.parametrize("raw,positive,expected", [
    ("not-harmful", "any", 0),
    ("partially-harmful", "any", 1),
    ("very-harmful", "any", 1),
    ("partially-harmful", "very", 0),
    ("very-harmful", "very", 1),
])
def test_binary_label(raw, positive, expected):
    assert _binary_label(raw, positive) == expected


def test_binary_label_rejects_unknown():
    with pytest.raises(ValueError):
        _binary_label("maybe", "any")


def test_convert_writes_loader_layout(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, "PNG")
    frame = pd.DataFrame({
        "id": ["a", "a"],  # repeated id must not overwrite an image
        "image": [{"bytes": buf.getvalue(), "path": None}] * 2,
        "text": ["hi", None],
        "label": ["not-harmful", "very-harmful"],
    })
    src = tmp_path / "src"
    src.mkdir()
    frame.to_parquet(src / "train-00000-of-00001.parquet")

    out = tmp_path / "out"
    summary = convert(str(src), str(out))
    assert summary["train"] == {"rows": 2, "positive": 1, "skipped": 0}

    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == [0, 1]
    assert rows[1]["text"] == ""
    assert len({r["img"] for r in rows}) == 2
    assert all(os.path.isfile(out / r["img"]) for r in rows)
