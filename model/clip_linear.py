"""CLIP image+text linear probe (frozen CLIP + logistic regression).

Features are ``[img, txt, img*txt]`` from L2-normalised CLIP ViT-B/32
embeddings, so the classifier can see the image, the text and how they
interact - which text-only rules cannot.  The trained weights are a small
``.npz`` file (a few KB) that is easy to commit and download.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

DEFAULT_PATH = os.path.join("model", "checkpoints", "clip_linear.npz")


def featurize(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """``[img, txt, img*txt]`` rows. Inputs are (N, D) arrays."""
    return np.concatenate([img, txt, img * txt], axis=1)


def clip_embed(model: Any, preprocess: Any, device: str, images: Sequence[Any], texts: Sequence[str]):
    """Normalised CLIP embeddings for PIL images and strings -> two (N, 512) arrays."""
    import clip
    import torch

    with torch.no_grad():
        x = torch.stack([preprocess(im) for im in images]).to(device)
        fi = model.encode_image(x)
        ft = model.encode_text(clip.tokenize([t or "" for t in texts], truncate=True).to(device))
        fi = fi / fi.norm(dim=-1, keepdim=True)
        ft = ft / ft.norm(dim=-1, keepdim=True)
    return fi.float().cpu().numpy(), ft.float().cpu().numpy()


def extract(samples: List[dict], embed_fn: Callable[[Sequence[Any], Sequence[str]], Any], batch: int = 64, log: bool = True):
    """Embed every sample (dicts with ``img`` path and ``text``). Returns (img, txt, labels)."""
    from PIL import Image

    I, T, Y = [], [], []
    for start in range(0, len(samples), batch):
        chunk = samples[start:start + batch]
        ims = [Image.open(s["img"]).convert("RGB") for s in chunk]
        fi, ft = embed_fn(ims, [s["text"] for s in chunk])
        I.append(fi); T.append(ft); Y += [int(s["label"]) for s in chunk]
        if log and (start // batch) % 10 == 0:
            print(f"[CHEM][clip_linear] embedded {min(start + batch, len(samples))}/{len(samples)}")
    return np.concatenate(I), np.concatenate(T), np.array(Y)


class ClipLinear:
    """Inference wrapper around the saved ``.npz`` weights."""

    def __init__(self, path: Optional[str] = None):
        path = path or os.environ.get("CHEM_CLIP_LINEAR_PATH") or DEFAULT_PATH
        z = np.load(path)
        self.coef, self.intercept = z["coef"].reshape(-1), float(z["intercept"])
        self.mean, self.std = z["mean"].reshape(-1), z["std"].reshape(-1)
        self.threshold = float(z["threshold"])

    def predict_proba(self, img: np.ndarray, txt: np.ndarray) -> np.ndarray:
        x = (featurize(img, txt) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept)))

    def to_score(self, p: float) -> float:
        """Map probability to the 0..1 pipeline scale so the dev threshold lands on 0.5."""
        t = min(max(self.threshold, 1e-3), 1 - 1e-3)
        return float(0.5 + (p - t) * (0.5 / max(t, 1 - t)))
