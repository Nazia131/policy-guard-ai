"""One uploaded image must produce exactly one image node."""
import torch
from PIL import Image

from encoders.image_encoder import ImageEncoder
from graph.graph_builder import build_graph
from utils.config import get_visual_concepts


class _StubModel:
    """Stand-in for CLIP: image features fixed, logits chosen by the test."""

    def __init__(self, probs):
        self.logit_scale = torch.tensor(0.0)
        self._probs = probs

    def encode_image(self, _t):
        return torch.ones(1, 512)


def _encoder(top_harmful: bool):
    enc = ImageEncoder.__new__(ImageEncoder)
    enc._torch = torch
    enc.device = "cpu"
    enc._min_conf = 0.25
    cfg = get_visual_concepts(None)
    enc._concepts = [{**c, "harmful": True} for c in cfg["harmful"]] + \
                    [{**c, "harmful": False} for c in cfg["safe"]]
    n = len(enc._concepts)
    # text features whose similarity to the (normalised) image gives the wanted softmax
    idx = 0 if top_harmful else n - 1
    tf = torch.zeros(n, 512)
    tf[:, 0] = 0.0
    tf[idx, 0] = 1.0
    enc._text_features = tf
    model = _StubModel(None)
    model.logit_scale = torch.tensor(6.0)  # exp(6) ~ 403 -> sharply peaked softmax
    enc._model = model
    enc._preprocess = lambda pil: torch.zeros(3, 4, 4)
    return enc


def test_harmful_image_yields_single_node():
    out = _encoder(True).encode(Image.new("RGB", (8, 8)))
    assert len(out["elements"]) == 1
    el = out["elements"][0]
    assert el["id"] == "image_0" and el["harmful_signal"] is True
    assert el["label"] == el["policy_category"]
    assert out["harm_score"] == el["concepts"][0]["confidence"] > 0.25


def test_safe_image_yields_single_safe_node():
    out = _encoder(False).encode(Image.new("RGB", (8, 8)))
    assert len(out["elements"]) == 1
    el = out["elements"][0]
    assert el["harmful_signal"] is False and el["label"] == "image"
    assert out["harm_score"] == 0.0 and el["concepts"] == []


def test_graph_has_one_image_node_and_no_image_image_edge():
    out = _encoder(True).encode(Image.new("RGB", (8, 8)))
    g = build_graph(out["elements"])
    assert len([n for n in g["nodes"] if n["type"] == "image"]) == 1
    assert g["edges"] == []
    assert g["nodes"][0]["concepts"]
