"""Tests for explicit GAT model status and checkpoint provenance.

Covers audit Phase 1, steps 1-2:

* a missing checkpoint is reported as ``untrained`` (never silently "fine");
* strict mode refuses to run without real trained weights;
* a saved checkpoint round-trips and its weights are genuinely loaded;
* legacy checkpoints still load but are reported as ``loaded_unverified``;
* checkpoints trained on the synthetic demo data are labelled ``synthetic_only``;
* the status reaches ``/health`` and the ``/analyze`` payload;
* ``model.train.train`` writes provenance metadata into the checkpoint.

These tests use tiny fake graphs and stub encoders: they check *engineering
correctness* of the status/provenance plumbing.  They say nothing about model
quality.

Run from the repository root: ``pytest tests/test_model_status.py``
"""

from __future__ import annotations

import json
import os
import random
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from model.checkpoint_io import (  # noqa: E402
    STATUS_SYNTHETIC,
    STATUS_TRAINED,
    STATUS_UNTRAINED,
    STATUS_UNVERIFIED,
    build_metadata,
    describe_architecture,
    read_checkpoint,
    save_checkpoint,
)
from model.gat_model import CHEMGraphAttentionNetwork  # noqa: E402
from model.harm_scorer import HarmScorer  # noqa: E402
from utils.provenance import SYNTHETIC_MARKER_FILENAME, is_synthetic_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _metadata(**overrides):
    """Build valid checkpoint metadata, overriding selected fields."""
    fields = dict(
        dataset_name="unit_test_dataset",
        data_dir=None,
        label_mapping="0 = safe, 1 = harmful (unit test)",
        is_synthetic=False,
        split_scheme="unit test",
        split_seed=1,
        split_counts={"train": 1, "val": 1, "test": 1},
        hyperparameters={"learning_rate": 0.1},
        architecture=describe_architecture(CHEMGraphAttentionNetwork()),
    )
    fields.update(overrides)
    return build_metadata(**fields)


def _save(tmp_path, metadata, name="ckpt.pt"):
    """Save a seeded model + metadata checkpoint; return (path, model)."""
    torch.manual_seed(123)
    model = CHEMGraphAttentionNetwork()
    path = tmp_path / name
    save_checkpoint(str(path), model.state_dict(), metadata, val_f1=0.5, epoch=3)
    return path, model


def _scorer(tmp_path, checkpoint=None, **kwargs):
    """Construct a HarmScorer pointing at ``checkpoint`` (or a missing file)."""
    model_path = str(checkpoint) if checkpoint else str(tmp_path / "missing.pt")
    return HarmScorer(
        model_path=model_path,
        library_path=str(tmp_path / "no_library.json"),
        device="cpu",
        **kwargs,
    )


def _tiny_graph():
    """A minimal graph dict accepted by HarmScorer.score()."""
    from torch_geometric.data import Data

    x = torch.randn(3, 768)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return {"data": Data(x=x, edge_index=edge_index), "networkx": None, "nodes": []}


# --------------------------------------------------------------------------- #
# Step 1: model status in HarmScorer
# --------------------------------------------------------------------------- #
def test_missing_checkpoint_is_reported_untrained(tmp_path):
    scorer = _scorer(tmp_path)
    assert scorer.weights_loaded is False
    assert scorer.model_status == STATUS_UNTRAINED
    info = scorer.model_info()
    assert info["model_status"] == STATUS_UNTRAINED
    assert info["weights_loaded"] is False
    assert info["checkpoint_path"] is None
    assert "randomly initialised" in info["warning"]
    assert scorer.model.training is False  # eval mode


def test_strict_mode_raises_without_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        _scorer(tmp_path, require_trained=True)


def test_checkpoint_round_trip_loads_the_saved_weights(tmp_path):
    path, saved_model = _save(tmp_path, _metadata())
    scorer = _scorer(tmp_path, path)

    assert scorer.weights_loaded is True
    assert scorer.model_status == STATUS_TRAINED
    assert scorer.model.training is False
    for key, value in saved_model.state_dict().items():
        assert torch.equal(scorer.model.state_dict()[key].cpu(), value), key

    info = scorer.model_info()
    assert info["dataset_name"] == "unit_test_dataset"
    assert info["is_synthetic"] is False
    assert info["val_f1"] == 0.5
    assert info["epoch"] == 3
    assert info["warning"] is None


def test_loaded_weights_differ_from_a_random_init(tmp_path):
    """Guard against the round-trip test passing trivially."""
    path, saved_model = _save(tmp_path, _metadata())
    untrained = _scorer(tmp_path)
    same = all(
        torch.equal(untrained.model.state_dict()[k].cpu(), v)
        for k, v in saved_model.state_dict().items()
    )
    assert not same


def test_synthetic_checkpoint_is_labelled_and_rejected_in_strict_mode(tmp_path):
    path, _ = _save(tmp_path, _metadata(is_synthetic=True, dataset_name="synthetic_demo"))
    scorer = _scorer(tmp_path, path)
    assert scorer.weights_loaded is True
    assert scorer.model_status == STATUS_SYNTHETIC
    assert scorer.model_info()["warning"]
    with pytest.raises(RuntimeError):
        _scorer(tmp_path, path, require_trained=True)


def test_legacy_checkpoint_formats_load_as_unverified(tmp_path):
    torch.manual_seed(5)
    model = CHEMGraphAttentionNetwork()

    bare = tmp_path / "bare.pt"
    torch.save(model.state_dict(), bare)

    wrapped = tmp_path / "wrapped.pt"
    torch.save({"state_dict": model.state_dict(), "val_f1": 0.4, "epoch": 2}, wrapped)

    for path in (bare, wrapped):
        scorer = _scorer(tmp_path, path)
        assert scorer.weights_loaded is True, path.name
        assert scorer.model_status == STATUS_UNVERIFIED, path.name
        assert scorer.model_info()["warning"], path.name
        # Unverified is allowed in strict mode (it is real weights, just no provenance).
        _scorer(tmp_path, path, require_trained=True)


def test_architecture_mismatch_raises_a_clear_error(tmp_path):
    arch = describe_architecture(CHEMGraphAttentionNetwork())
    arch["hidden_channels"] = 128
    path, _ = _save(tmp_path, _metadata(architecture=arch))
    with pytest.raises(ValueError, match="hidden_channels"):
        _scorer(tmp_path, path)


def test_score_output_carries_model_status(tmp_path):
    result = _scorer(tmp_path).score(_tiny_graph())
    assert result["model_status"] == STATUS_UNTRAINED


# --------------------------------------------------------------------------- #
# Step 2: checkpoint metadata
# --------------------------------------------------------------------------- #
def test_metadata_stores_no_absolute_local_path(tmp_path):
    data_dir = tmp_path / "my_dataset"
    data_dir.mkdir()
    meta = _metadata(data_dir=str(data_dir))
    assert meta["dataset_dir_name"] == "my_dataset"
    assert str(tmp_path) not in json.dumps(meta)


def test_metadata_survives_weights_only_load(tmp_path):
    path, _ = _save(tmp_path, _metadata())
    _, meta = read_checkpoint(str(path), "cpu")
    for key in (
        "format_version",
        "dataset_name",
        "is_synthetic",
        "label_mapping",
        "split_scheme",
        "split_seed",
        "split_counts",
        "hyperparameters",
        "architecture",
        "torch_version",
        "created_utc",
    ):
        assert key in meta, key
    assert meta["val_f1"] == 0.5 and meta["epoch"] == 3


def test_synthetic_generator_writes_marker(tmp_path):
    from data.create_synthetic_demo import create_synthetic_demo

    out = tmp_path / "syn"
    create_synthetic_demo(n_samples=20, out_dir=str(out))
    assert (out / SYNTHETIC_MARKER_FILENAME).is_file()
    assert is_synthetic_dir(str(out)) is True
    assert is_synthetic_dir(str(tmp_path)) is False
    assert is_synthetic_dir(None) is False


# --------------------------------------------------------------------------- #
# Training writes provenance (stub encoders; checks plumbing, not accuracy)
# --------------------------------------------------------------------------- #
class _StubTextEncoder:
    """Deterministic stand-in for TextEncoder (no model download)."""

    def __init__(self, device="cpu"):
        pass

    def encode(self, text):
        rng = random.Random(text)
        return {
            "elements": [
                {
                    "id": "t0",
                    "type": "text_claim",
                    "modality": "text",
                    "label": "stub",
                    "confidence": 0.9,
                    "harmful_signal": False,
                    "embedding": [rng.uniform(-1, 1) for _ in range(768)],
                }
            ],
            "harm_score": 0.0,
        }


class _StubImageEncoder:
    """Deterministic stand-in for ImageEncoder (no model download)."""

    def __init__(self, device="cpu"):
        pass

    def encode(self, image):
        rng = random.Random(str(image.getpixel((0, 0))))
        return {
            "elements": [
                {
                    "id": "i0",
                    "type": "image_classification",
                    "modality": "image",
                    "label": "stub",
                    "confidence": 0.8,
                    "harmful_signal": False,
                    "embedding": [rng.uniform(-1, 1) for _ in range(768)],
                }
            ],
            "harm_score": 0.0,
        }


def _run_training(tmp_path, monkeypatch, data_dir, **train_kwargs):
    """Run model.train.train with stub encoders and a temp checkpoint path."""
    import model.train as train_module

    text_mod = types.ModuleType("encoders.text_encoder")
    text_mod.TextEncoder = _StubTextEncoder
    image_mod = types.ModuleType("encoders.image_encoder")
    image_mod.ImageEncoder = _StubImageEncoder
    monkeypatch.setitem(sys.modules, "encoders.text_encoder", text_mod)
    monkeypatch.setitem(sys.modules, "encoders.image_encoder", image_mod)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_path = ckpt_dir / "best_gat.pt"
    monkeypatch.setattr(train_module, "_CHECKPOINT_DIR", str(ckpt_dir))
    monkeypatch.setattr(train_module, "_BEST_CHECKPOINT_PATH", str(ckpt_path))
    monkeypatch.setenv("WANDB_MODE", "disabled")

    result = train_module.train(
        epochs=2, batch_size=8, data_dir=str(data_dir), **train_kwargs
    )
    assert result == str(ckpt_path)
    assert ckpt_path.is_file()
    return ckpt_path


def test_training_on_synthetic_data_writes_synthetic_provenance(tmp_path, monkeypatch):
    from data.create_synthetic_demo import create_synthetic_demo

    data_dir = tmp_path / "syn"
    create_synthetic_demo(n_samples=20, out_dir=str(data_dir))
    ckpt = _run_training(tmp_path, monkeypatch, data_dir)

    _, meta = read_checkpoint(str(ckpt), "cpu")
    assert meta["is_synthetic"] is True
    assert meta["dataset_name"] == "synthetic_demo"
    assert meta["format_version"] == 2
    # train.py loads the train + dev files only (the test file is not used).
    rows = sum(
        sum(1 for _ in open(data_dir / name, encoding="utf-8"))
        for name in ("train.jsonl", "dev.jsonl")
    )
    assert sum(meta["split_counts"].values()) == rows
    assert meta["hyperparameters"]["batch_size"] == 8
    assert meta["architecture"] == describe_architecture(CHEMGraphAttentionNetwork())

    scorer = _scorer(tmp_path, ckpt)
    assert scorer.model_status == STATUS_SYNTHETIC
    with pytest.raises(RuntimeError):
        _scorer(tmp_path, ckpt, require_trained=True)


def test_training_records_the_dataset_name_and_label_mapping(tmp_path, monkeypatch):
    from data.create_synthetic_demo import create_synthetic_demo

    data_dir = tmp_path / "pretend_real"
    create_synthetic_demo(n_samples=20, out_dir=str(data_dir))
    os.remove(data_dir / SYNTHETIC_MARKER_FILENAME)  # simulate a non-synthetic dir

    ckpt = _run_training(
        tmp_path,
        monkeypatch,
        data_dir,
        dataset_name="unit_test_stub_dataset",
        label_mapping="0 = not harmful, 1 = harmful (stub mapping)",
    )
    _, meta = read_checkpoint(str(ckpt), "cpu")
    assert meta["is_synthetic"] is False
    assert meta["dataset_name"] == "unit_test_stub_dataset"
    assert meta["label_mapping"] == "0 = not harmful, 1 = harmful (stub mapping)"
    assert _scorer(tmp_path, ckpt).model_status == STATUS_TRAINED


# --------------------------------------------------------------------------- #
# API: status reaches /health and the /analyze payload
# --------------------------------------------------------------------------- #
def _api_pipeline(scorer):
    """A CHEMPipeline wired to a real scorer + graph builder and a stub text encoder."""
    pytest.importorskip("fastapi")
    from api.app import CHEMPipeline
    from graph.graph_builder import ContentCompositionGraphBuilder
    from model.harm_scorer import fuse_modality_scores

    pipe = CHEMPipeline()
    pipe.loaded = True
    pipe.harm_scorer = scorer
    pipe.graph_builder = ContentCompositionGraphBuilder()
    pipe.text_encoder = _StubTextEncoder()
    pipe._fuse = fuse_modality_scores
    return pipe


def test_env_flag_parsing(monkeypatch):
    pytest.importorskip("fastapi")
    from api.app import _env_flag

    monkeypatch.delenv("CHEM_TEST_FLAG", raising=False)
    assert _env_flag("CHEM_TEST_FLAG") is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CHEM_TEST_FLAG", value)
        assert _env_flag("CHEM_TEST_FLAG") is True, value
    for value in ("", "0", "false", "no"):
        monkeypatch.setenv("CHEM_TEST_FLAG", value)
        assert _env_flag("CHEM_TEST_FLAG") is False, value


def test_health_reports_degraded_for_untrained_model(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import api.app as api_app

    monkeypatch.setattr(api_app, "pipeline", _api_pipeline(_scorer(tmp_path)))
    body = api_app.health()
    assert body["models_loaded"] is True
    assert body["status"] == "degraded"
    assert body["model"]["model_status"] == STATUS_UNTRAINED
    assert body["model"]["warning"]


def test_health_reports_ok_only_for_a_trained_model(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import api.app as api_app

    path, _ = _save(tmp_path, _metadata())
    monkeypatch.setattr(api_app, "pipeline", _api_pipeline(_scorer(tmp_path, path)))
    body = api_app.health()
    assert body["status"] == "ok"
    assert body["model"]["model_status"] == STATUS_TRAINED
    assert body["model"]["dataset_name"] == "unit_test_dataset"


def test_health_reports_not_loaded_when_scorer_missing(monkeypatch):
    pytest.importorskip("fastapi")
    import api.app as api_app

    monkeypatch.setattr(api_app, "pipeline", api_app.CHEMPipeline())
    body = api_app.health()
    assert body["status"] == "degraded"
    assert body["model"]["model_status"] == "not_loaded"


def test_analyze_payload_carries_model_status(tmp_path):
    untrained = _api_pipeline(_scorer(tmp_path))
    for result in (untrained.analyze(), untrained.analyze(text="a calm afternoon")):
        assert result["model_status"] == STATUS_UNTRAINED
        assert result["model_warning"]

    path, _ = _save(tmp_path, _metadata())
    trained = _api_pipeline(_scorer(tmp_path, path))
    result = trained.analyze(text="a calm afternoon")
    assert result["model_status"] == STATUS_TRAINED
    assert result["model_warning"] is None
