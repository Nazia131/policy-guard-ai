"""The API explains HARMFUL and borderline-SAFE verdicts, but skips clearly-safe ones."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from api.app import _should_explain  # noqa: E402


def test_harmful_is_always_explained():
    assert _should_explain(True, 0.95, 0.5)
    assert _should_explain(True, 0.51, 0.5)


def test_borderline_safe_is_explained():
    # 0.45 < 0.5 so SAFE, but within the 0.1 margin -> a reviewer should see why.
    assert _should_explain(False, 0.45, 0.5)


def test_clearly_safe_is_not_explained():
    assert not _should_explain(False, 0.10, 0.5)
    assert not _should_explain(False, 0.39, 0.5)
