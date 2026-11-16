import math
import struct
import sys
import types
import wave

import pytest

pytest.importorskip("librosa")


def _write_wav(path, seconds=2.0, sr=16000, freq=220.0):
    n = int(seconds * sr)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / sr))) for i in range(n)))


class _FakeWhisper:
    def __init__(self, text): self.text = text; self.got = None
    def transcribe(self, audio, **kw):
        self.got = audio
        return {"text": self.text, "language": "en"}


def _encoder(monkeypatch, text):
    fake = types.SimpleNamespace(load_model=lambda *a, **k: _FakeWhisper(text))
    monkeypatch.setitem(sys.modules, "whisper", fake)
    from encoders.audio_encoder import AudioEncoder
    from rules.rule_engine import RuleEngine
    return AudioEncoder(text_encoder=RuleEngine(), device="cpu")


def test_whisper_receives_waveform_not_a_path(monkeypatch, tmp_path):
    enc = _encoder(monkeypatch, "hello there")
    f = tmp_path / "a.wav"; _write_wav(f)
    out = enc.encode(str(f))
    assert not isinstance(enc._whisper.got, str)
    assert getattr(enc._whisper.got, "dtype", None) == "float32"
    assert out["transcript"] == "hello there"


def test_agitation_alone_does_not_make_audio_harmful(monkeypatch, tmp_path):
    enc = _encoder(monkeypatch, "what a lovely day")
    f = tmp_path / "a.wav"; _write_wav(f)
    out = enc.encode(str(f))
    assert out["harm_score"] == 0.0


def test_harmful_transcript_is_scored(monkeypatch, tmp_path):
    enc = _encoder(monkeypatch, "i will kill you")
    f = tmp_path / "a.wav"; _write_wav(f)
    assert enc.encode(str(f))["harm_score"] >= 0.5
