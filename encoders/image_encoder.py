"""CHEM image encoder.

Wraps OpenAI CLIP (``ViT-B/32``) to perform zero-shot classification of an
image against the full set of policy-defined visual concepts (both harmful and
safe).  The safe concepts anchor the softmax so that harmful scores are always
relative to plausible benign interpretations.

No harm categories, prompts, or thresholds are hardcoded: the concept prompts
come from ``visual_concepts`` in ``policy/policy_config.json`` and the decision
threshold from the ``image_harmful_min_confidence`` coefficient.

Public API
----------
``ImageEncoder(config_path=None, device=None)``
``ImageEncoder.encode(image) -> dict`` where ``image`` is a PIL.Image or a
base64-encoded string.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

from utils.config import get_coefficient, get_policy_clause, get_visual_concepts
from utils.embeddings import to_fixed_dim


class ImageEncoder:
    """Encode an image into CHEM elements via CLIP zero-shot classification.

    CLIP and the tokenised concept prompts are prepared once in
    :meth:`__init__`; :meth:`encode` only runs cheap forward passes.

    Parameters
    ----------
    config_path : str, optional
        Override path to ``policy_config.json``.
    device : str, optional
        Torch device string; auto-selected (CUDA if available else CPU) when
        omitted.

    Attributes
    ----------
    modality : str
        Always ``"image"``.
    """

    #: This encoder's fixed modality tag.
    modality = "image"

    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """Load CLIP ``ViT-B/32`` once and pre-tokenise all concept prompts.

        Inputs: optional config path and device.
        Outputs: a ready ``ImageEncoder`` with CLIP resident and the harmful +
        safe concept prompt embeddings cached.
        """
        import clip  # openai CLIP package
        import torch

        self._config_path = config_path
        self._torch = torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[CHEM][image_encoder] Loading CLIP ViT-B/32 on device={self.device} ...")
        self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
        self._model.eval()

        # Decision threshold for calling the top concept "harmful".
        self._min_conf = float(get_coefficient("image_harmful_min_confidence", config_path))

        # Build the ordered concept table: (prompt, category, harmful_flag).
        concepts = get_visual_concepts(config_path)
        self._concepts: List[Dict[str, Any]] = []
        for entry in concepts.get("harmful", []):
            self._concepts.append({**entry, "harmful": True})
        for entry in concepts.get("safe", []):
            self._concepts.append({**entry, "harmful": False})

        if not self._concepts:
            raise ValueError("[CHEM][image_encoder] No visual_concepts found in config.")

        # Pre-tokenise and pre-encode the text side of CLIP once — the concept
        # set is fixed, so this never needs to be recomputed per image.
        prompts = [c["prompt"] for c in self._concepts]
        tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self._model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features  # [C, D]

        print(
            f"[CHEM][image_encoder] Ready. {len(self._concepts)} concept(s) "
            f"({sum(c['harmful'] for c in self._concepts)} harmful), "
            f"min_confidence={self._min_conf}"
        )

    # ------------------------------------------------------------------ utils
    def _to_pil(self, image: Any):
        """Coerce ``image`` into a RGB PIL.Image.

        Accepts an existing PIL.Image or a base64-encoded string (with or
        without a ``data:`` URI prefix).

        Inputs: PIL.Image or base64 str.
        Outputs: an ``RGB`` PIL.Image.
        """
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            # Strip an optional data-URI header ("data:image/png;base64,....").
            payload = image.split(",", 1)[-1] if image.startswith("data:") else image
            raw = base64.b64decode(payload)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        raise TypeError(
            f"[CHEM][image_encoder] Unsupported image type: {type(image)!r}; "
            "expected PIL.Image or base64 str."
        )

    def embed_image(self, image: Any) -> List[float]:
        """Return a 768-d (padded) CLIP image embedding.

        Exposed so other encoders (e.g. the video encoder) can obtain a raw
        image feature without running the full zero-shot classification.

        Inputs: PIL.Image or base64 str.
        Outputs: list[float] length 768 (CLIP is 512-d, zero-padded).
        """
        torch = self._torch
        pil = self._to_pil(image)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return to_fixed_dim(feats.squeeze(0).detach().cpu().tolist())

    # ------------------------------------------------------------------ encode
    def encode(self, image: Any) -> Dict[str, Any]:
        """Zero-shot classify ``image`` against all policy visual concepts.

        The image embedding is compared (softmax over cosine similarities)
        against every harmful and safe concept prompt.  ``harmful_signal`` is
        True only when the single top-scoring concept is a harmful one AND its
        score exceeds ``image_harmful_min_confidence``.  ``harm_score`` is the
        highest harmful-concept score that cleared that threshold, else 0.0.

        Inputs
        ------
        image : PIL.Image | str
            An image object or a base64-encoded image string.

        Outputs
        -------
        dict
            ``{"elements": [...], "harm_score": float, "modality": "image"}``.
        """
        torch = self._torch
        print("[CHEM][image_encoder] encode() ...")

        try:
            pil = self._to_pil(image)
        except Exception as exc:  # graceful failure -> empty, non-harmful result
            print(f"[CHEM][image_encoder] Failed to decode image: {exc}")
            return {"elements": [], "harm_score": 0.0, "modality": self.modality}

        # --- CLIP image feature + similarity distribution over concepts ------
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self._model.encode_image(tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            # CLIP's learned temperature scaling before softmax.
            logits = self._model.logit_scale.exp() * img_features @ self._text_features.t()
            probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().tolist()

        base_embedding = to_fixed_dim(img_features.squeeze(0).detach().cpu().tolist())

        # Identify the overall top concept.
        top_i = max(range(len(probs)), key=lambda i: probs[i])
        top_concept = self._concepts[top_i]
        top_score = float(probs[top_i])
        top_is_harmful = bool(top_concept["harmful"]) and top_score > self._min_conf

        # --- Build elements --------------------------------------------------
        elements: List[Dict[str, Any]] = []
        idx = 0

        # Base image node carrying the CLIP visual embedding.
        elements.append(
            {
                "id": f"image_{idx}",
                "label": "image",
                "type": "image",
                "embedding": base_embedding,
                "confidence": top_score,
                "harmful_signal": top_is_harmful,
                "full_label": top_concept["prompt"],
                "modality": self.modality,
                "policy_category": top_concept["category"],
            }
        )
        idx += 1

        # Emit a dedicated node for every harmful concept clearing the
        # threshold, tracking the max qualifying harmful score for harm_score.
        harm_score = 0.0
        for concept, prob in zip(self._concepts, probs):
            if not concept["harmful"]:
                continue
            if prob <= self._min_conf:
                continue
            harm_score = max(harm_score, float(prob))
            clause = get_policy_clause(concept["category"], self._config_path)
            elements.append(
                {
                    "id": f"image_{idx}",
                    "label": concept["category"],
                    "type": "image",
                    "embedding": base_embedding,
                    "confidence": float(prob),
                    # Only the actual top harmful concept is the driving signal;
                    # other threshold-clearing concepts are informative context.
                    "harmful_signal": bool(
                        top_is_harmful and self._concepts[top_i] is concept
                    ),
                    "full_label": f"{clause.get('title', concept['category'])}: {concept['prompt']}",
                    "modality": self.modality,
                    "policy_category": concept["category"],
                }
            )
            idx += 1

        print(
            f"[CHEM][image_encoder] -> top='{top_concept['prompt'][:48]}' "
            f"score={top_score:.4f} harmful={top_is_harmful} "
            f"harm_score={harm_score:.4f} elements={len(elements)}"
        )
        return {
            "elements": elements,
            "harm_score": float(harm_score),
            "modality": self.modality,
        }
