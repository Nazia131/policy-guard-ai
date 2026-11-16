import io
import os
import sys
import types

import pytest

from rules.rule_engine import RuleEngine


@pytest.fixture(scope="module")
def eng():
    return RuleEngine()


HARMFUL = [
    "Chinese people are spreading the virus on purpose",
    "kill all the jews",
    "you should kill yourself",
    "i will kill you",
    "Italians are lazy and dirty",
    "go back to china with your virus",
    "The government says 5G causes covid",
    "N1gg3r",
    "they should all be shot",
    "I know where you live",
]
SAFE = [
    "I got my second shot today, feeling great",
    "We're killing it at the gym",
    "Stop Asian hate! Chinese people are not the virus",
    "When you finally get the weekend off",
    "Police said a bomb threat was reported at the school",
    "I love my dog",
    "The chef chopped the onions with a kitchen knife",
    "Wash your hands and wear a mask",
    "Execute the script and burn calories at the gym",
    "88 % of people got vaccinated",
]


@pytest.mark.parametrize("text", HARMFUL)
def test_harmful_examples_flagged(eng, text):
    assert eng.analyze(text)["score"] >= eng.threshold, text


@pytest.mark.parametrize("text", SAFE)
def test_benign_examples_not_flagged(eng, text):
    assert eng.analyze(text)["score"] < eng.threshold, text


def test_empty_text_scores_zero(eng):
    assert eng.analyze("")["score"] == 0.0
    assert eng.analyze(None)["score"] == 0.0


def test_encode_matches_text_encoder_shape(eng):
    out = eng.encode("kill all the jews")
    assert set(out) >= {"elements", "harm_score", "modality"}
    assert out["elements"][0]["label"] == "text"
    assert any(e.get("harmful_signal") for e in out["elements"])


def test_child_safety_not_dampened_by_news_context(eng):
    plain = eng.analyze("child porn links")["score"]
    news = eng.analyze("police said child porn links were found, according to reports")["score"]
    assert news == plain


def test_rules_json_every_pattern_compiles(eng):
    assert len(eng._rules) == len(eng.rules['lexicon']) + len(eng.rules['patterns'])


def test_api_rules_mode_end_to_end(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("CHEM_MODE", "rules")
    from fastapi.testclient import TestClient
    from api import app as appmod

    appmod.pipeline = appmod.CHEMPipeline()
    appmod.pipeline.load()
    appmod.pipeline._rules.image_encoder = None
    appmod.pipeline._rules.audio_encoder = None
    with TestClient(appmod.app) as c:
        appmod.pipeline.load()
        appmod.pipeline._rules.image_encoder = None
        appmod.pipeline._rules.audio_encoder = None
        bad = c.post("/analyze", data={"text": "kill all the jews"}).json()
        ok = c.post("/analyze", data={"text": "got my second shot today"}).json()
    assert bad["verdict"] == "HARMFUL" and bad["policy_clause"]["clause_id"]
    assert bad["triggered_elements"]
    assert ok["verdict"] == "SAFE" and ok["policy_clause"] is None and ok["triggered_elements"] == []
    assert "No policy violation" in ok["explanation"]
