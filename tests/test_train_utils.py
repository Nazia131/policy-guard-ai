"""Class weights, threshold tuning, graph cache and an end-to-end train() run
(with stub encoders, so no model downloads)."""
import json
import os

import pytest
import torch

from model import train as train_mod
from model import train_utils as tu


def test_class_weights_inverse_frequency():
    w = tu.class_weights([0] * 75 + [1] * 25)
    assert w == pytest.approx([100 / 150, 100 / 50])
    assert tu.class_weights([0, 0]) == pytest.approx([0.5, 1.0])  # missing class -> 1.0


def test_best_threshold_finds_separating_cut_and_is_validation_only():
    y = [0] * 50 + [1] * 50
    p = [0.2] * 50 + [0.7] * 50  # perfectly separable between 0.2 and 0.7
    t = tu.best_threshold(y, p, "f1")
    assert 0.2 < t <= 0.7
    assert tu.metrics_at_threshold(y, p, t)["accuracy"] == 1.0
    with pytest.raises(ValueError):
        tu.best_threshold(y, p, "auc")


def test_threshold_beats_default_when_scores_are_shifted():
    y = [0] * 60 + [1] * 40
    p = [0.1] * 60 + [0.4] * 40  # all below 0.5 -> default predicts all safe
    assert tu.metrics_at_threshold(y, p, 0.5)["recall"] == 0.0
    t = tu.best_threshold(y, p, "f1")
    assert tu.metrics_at_threshold(y, p, t)["f1"] == 1.0


def test_subsample_is_deterministic_and_keeps_order():
    s = [{"id": i} for i in range(100)]
    a, b = tu.subsample(s, 10), tu.subsample(s, 10)
    assert a == b and len(a) == 10 and a == sorted(a, key=lambda r: r["id"])
    assert tu.subsample(s, None) is s and tu.subsample(s, 500) is s


def test_cache_key_changes_with_samples_and_source(tmp_path):
    s1 = [{"id": 1, "img": "a.png", "text": "x", "label": 0}]
    s2 = [{"id": 1, "img": "a.png", "text": "x", "label": 1}]
    assert tu.graph_cache_key(s1) == tu.graph_cache_key(s1)
    assert tu.graph_cache_key(s1) != tu.graph_cache_key(s2)
    # editing an encoder file must invalidate the cache
    root = tmp_path
    for rel in tu._GRAPH_SHAPING_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("v1")
    k1 = tu.graph_cache_key(s1, str(root))
    (root / "encoders/image_encoder.py").write_text("v2")
    assert tu.graph_cache_key(s1, str(root)) != k1


def test_cache_roundtrip_and_disabled(tmp_path):
    assert tu.load_graph_cache(str(tmp_path), "k") is None
    assert tu.load_graph_cache(None, "k") is None
    tu.save_graph_cache(str(tmp_path), "k", [{"a": 1}])
    assert tu.load_graph_cache(str(tmp_path), "k") == [{"a": 1}]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# ------------------------------------------------------------ end to end
class _StubText:
    def __init__(self, device=None):
        _StubText.created += 1

    def encode(self, text):
        harmful = "bad" in text
        return {"elements": [{
            "id": "text_0", "label": "hate" if harmful else "text", "type": "text",
            "modality": "text", "embedding": [1.0 if harmful else -1.0] + [0.0] * 767,
            "confidence": 0.9, "harmful_signal": harmful, "full_label": text}]}


class _StubImage:
    def __init__(self, device=None):
        pass

    def encode(self, image):
        return {"elements": []}


def _samples(tmp_path, n=40):
    from PIL import Image

    out = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4)).save(p)
        out.append({"id": i, "img": str(p), "text": "bad" if i % 3 == 0 else "ok", "label": int(i % 3 == 0)})
    return out


def test_train_end_to_end_uses_cache_weights_and_threshold(tmp_path, monkeypatch, capsys):
    samples = _samples(tmp_path)
    ckpt = tmp_path / "ckpt" / "best_gat.pt"
    monkeypatch.setattr(train_mod, "_CHECKPOINT_DIR", str(ckpt.parent))
    monkeypatch.setattr(train_mod, "_BEST_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setattr(train_mod, "_load_samples", lambda split, d: list(samples) if split == "train" else [])
    monkeypatch.setattr(train_mod, "_init_wandb", lambda *a, **k: None)
    monkeypatch.setattr("encoders.text_encoder.TextEncoder", _StubText)
    monkeypatch.setattr("encoders.image_encoder.ImageEncoder", _StubImage)
    _StubText.created = 0
    kw = dict(epochs=2, batch_size=8, cache_dir=str(tmp_path / "cache"))

    assert train_mod.train(**kw, max_samples=30) == str(ckpt)
    out = capsys.readouterr().out
    assert "using 30 samples" in out
    assert "class weights" in out
    assert "tuned on val f1" in out and "baseline : always-SAFE" in out
    assert _StubText.created == 1
    thr = json.loads((ckpt.parent / "best_gat.threshold.json").read_text())
    assert thr["chosen_on"] == "validation" and 0.05 <= thr["threshold"] <= 0.95

    # second run: same samples -> cache hit, encoders not constructed again
    train_mod.train(**kw, max_samples=30)
    assert "cached graphs" in capsys.readouterr().out
    assert _StubText.created == 1

    # --no-class-weights path still trains
    train_mod.train(**kw, max_samples=30, use_class_weights=False)
    assert "class weights" not in capsys.readouterr().out
