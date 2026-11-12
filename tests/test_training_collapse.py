"""Guards against the 'label everything harmful' collapse in model/train.py."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from model import train as T  # noqa: E402

_EDGES = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])


def _graphs(n: int, harmful_share: float, seed: int = 0, signal: float = 0.0):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        y = int(rng.random() < harmful_share)
        x = torch.randn(2, 768, generator=g)
        if y and signal:
            x[0, :8] += signal
        out.append(Data(x=x, edge_index=_EDGES, y=torch.tensor([y])))
    return out


# --- splitting ---------------------------------------------------------------


def test_split_is_stratified_and_partitions_everything():
    data = _graphs(1000, 0.85, seed=1)
    splits = T._split_dataset(data)
    assert sum(len(v) for v in splits.values()) == len(data)
    overall = sum(T._graph_label(g) for g in data) / len(data)
    for part in splits.values():
        share = sum(T._graph_label(g) for g in part) / len(part)
        assert share == pytest.approx(overall, abs=0.01)


def test_split_is_deterministic():
    data = _graphs(200, 0.6, seed=2)
    a = T._split_dataset(data)
    b = T._split_dataset(data)
    assert [id(g) for g in a["test"]] == [id(g) for g in b["test"]]


# --- class weights -----------------------------------------------------------


def test_class_weights_upweight_the_minority():
    w = T._class_weights(_graphs(1000, 0.85, seed=3))
    assert w[0] > w[1]  # safe is the minority here
    assert w[0].item() == pytest.approx(1000 / (2 * 0.15 * 1000), rel=0.2)


def test_class_weights_fall_back_when_a_class_is_missing():
    assert T._class_weights(_graphs(50, 1.0)).tolist() == [1.0, 1.0]


# --- collapse statistics -----------------------------------------------------


def test_constant_predictor_has_no_skill_in_balanced_metrics():
    y = [1] * 85 + [0] * 15
    stats = T._prediction_stats(y, [0.9] * 100)
    assert stats["pred_pos_rate"] == 1.0
    assert stats["accuracy"] == pytest.approx(0.85)
    assert stats["majority_acc"] == pytest.approx(0.85)  # no better than guessing
    assert stats["balanced_accuracy"] == pytest.approx(0.5)
    assert stats["macro_f1"] <= 0.5
    assert T._is_collapsed(stats)


def test_real_model_is_not_flagged_as_collapsed():
    y = [1, 1, 1, 0, 0, 0]
    stats = T._prediction_stats(y, [0.9, 0.8, 0.4, 0.2, 0.1, 0.7])
    assert not T._is_collapsed(stats)
    assert stats["balanced_accuracy"] > 0.5


def test_f1_selection_prefers_the_constant_predictor_but_auc_does_not():
    """The core bug: when harmful is the majority, F1 picks 'everything harmful'."""
    y = [1] * 85 + [0] * 15
    constant = [0.9] * 100
    real = [0.8] * 70 + [0.3] * 15 + [0.2] * 10 + [0.7] * 5  # ranks mostly right
    m_const = T._compute_metrics(y, constant)
    m_real = T._compute_metrics(y, real)
    s_const, s_real = T._prediction_stats(y, constant), T._prediction_stats(y, real)

    f1_const = T._selection_score(m_const, s_const, "f1")
    f1_real = T._selection_score(m_real, s_real, "f1")
    assert f1_const >= f1_real  # F1 happily rewards the constant predictor

    assert T._selection_score(m_const, s_const, "auc") == pytest.approx(0.5)
    assert T._selection_score(m_real, s_real, "auc") > 0.7


def test_default_selection_metric_is_not_f1():
    assert T._DEFAULT_SELECT_BY == "auc"


def test_best_threshold_recentres_a_biased_model():
    y = [1] * 50 + [0] * 50
    probs = [0.30] * 50 + [0.10] * 50  # ranks perfectly, but nothing exceeds 0.5
    thr, bal = T._best_threshold(y, probs)
    assert 0.10 < thr <= 0.30
    assert bal == pytest.approx(1.0)
    assert T._prediction_stats(y, probs)["balanced_accuracy"] == pytest.approx(0.5)


# --- end-to-end on a tiny synthetic run --------------------------------------


def test_fit_runs_and_saves_a_complete_checkpoint(tmp_path):
    splits = T._split_dataset(_graphs(90, 0.7, seed=4, signal=2.0))
    ckpt = tmp_path / "best.pt"
    result = T._fit(splits, epochs=3, batch_size=16, seed=0, checkpoint_path=str(ckpt))

    assert ckpt.is_file()
    saved = torch.load(str(ckpt), map_location="cpu")
    for key in ("state_dict", "val_f1", "val_score", "select_by", "threshold", "epoch"):
        assert key in saved
    assert saved["select_by"] == "auc"
    assert 0.0 < saved["threshold"] < 1.0
    for key in ("test_metrics", "test_stats", "test_stats_tuned", "learned_signal", "beats_majority"):
        assert key in result


def test_fit_is_reproducible_for_a_fixed_seed(tmp_path):
    splits = T._split_dataset(_graphs(60, 0.7, seed=5, signal=2.0))
    a = T._fit(splits, epochs=2, batch_size=16, seed=7, checkpoint_path=str(tmp_path / "a.pt"))
    b = T._fit(splits, epochs=2, batch_size=16, seed=7, checkpoint_path=str(tmp_path / "b.pt"))
    assert a["test_metrics"]["auc"] == pytest.approx(b["test_metrics"]["auc"])


def test_fit_rejects_an_unknown_selection_metric(tmp_path):
    splits = T._split_dataset(_graphs(30, 0.5, seed=6))
    with pytest.raises(ValueError):
        T._fit(splits, epochs=1, select_by="vibes", checkpoint_path=str(tmp_path / "x.pt"))
