"""Tests for the Hateful Memes loader (no torch / model stack required)."""

import json
import os

import pytest

from data.preprocess.hateful_memes import load_hateful_memes, load_samples


def _write_split(root, name, rows):
    """Write ``rows`` as JSON lines into ``root/name``."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


ROWS = [
    {"id": 1, "img": "img/1.png", "text": "hello", "label": 0},
    {"id": 2, "img": "img/2.png", "text": "world", "label": 1},
]


def test_load_resolves_absolute_image_paths(tmp_path):
    _write_split(tmp_path, "train.jsonl", ROWS)
    samples = list(load_hateful_memes(str(tmp_path), "train"))
    assert [s["label"] for s in samples] == [0, 1]
    assert all(os.path.isabs(s["img"]) for s in samples)


def test_load_finds_nested_layout(tmp_path):
    nested = tmp_path / "hateful_memes" / "hateful_memes"
    _write_split(nested, "dev_seen.jsonl", ROWS)
    samples = list(load_hateful_memes(str(tmp_path), "dev"))
    assert len(samples) == 2


def test_load_samples_uses_env_default(tmp_path, monkeypatch):
    _write_split(tmp_path, "test.jsonl", [{"id": 3, "img": "img/3.png", "text": "x"}])
    monkeypatch.setenv("CHEM_DATA_DIR", str(tmp_path))
    samples = list(load_samples("test"))
    assert samples[0]["label"] == -1  # blind split sentinel


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_hateful_memes(str(tmp_path / "nope"), "train"))
