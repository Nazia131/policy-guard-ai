import numpy as np
import pytest

pytest.importorskip("sklearn")

from evaluate.clip_linear_train import fit_and_report
from model.clip_linear import ClipLinear


def _split(rng, n):
    y = rng.integers(0, 2, n)
    i = rng.normal(size=(n, 16)); t = rng.normal(size=(n, 16))
    i[:, 0] += 1.5 * (2 * y - 1)  # signal lives in the image part
    return i, t, y


def test_probe_learns_planted_signal_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    out = str(tmp_path / "m.npz")
    m = fit_and_report(_split(rng, 600), _split(rng, 200), _split(rng, 300), out)
    assert m["auc"] > 0.8
    probe = ClipLinear(out)
    i, t, y = _split(np.random.default_rng(1), 100)
    p = probe.predict_proba(i, t)
    assert p.shape == (100,) and ((p >= probe.threshold) == y).mean() > 0.7
    assert abs(probe.to_score(probe.threshold) - 0.5) < 1e-9
