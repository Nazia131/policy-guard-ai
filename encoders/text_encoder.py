"""CHEM text encoder.

Turns a raw text string into CHEM *elements* (canonical node dicts) plus a
scalar text harm score.  Two frozen HuggingFace models are loaded once per
process:

* ``roberta-base`` -- provides a dense 768-d semantic embedding used as the
  node feature for the base text node (and as the feature for every derived
  policy-category node).
* ``unitary/toxic-bert`` -- a 6-label toxicity classifier whose outputs are
  re-expressed as CHEM policy categories via ``toxic_bert_mapping`` in
  ``policy/policy_config.json``.  No harm semantics or thresholds are hardcoded
  here; everything is read from the policy config.

Public API
----------
``TextEncoder(config_path=None, device=None)``
``TextEncoder.encode(text: str) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.config import get_policy_clause, load_policy_config
from utils.embeddings import to_fixed_dim


class TextEncoder:
    """Encode text into CHEM elements and a text harm score.

    The encoder loads its two transformer models exactly once (in
    :meth:`__init__`) and reuses them for every call to :meth:`encode`.

    Parameters
    ----------
    config_path : str, optional
        Override path to ``policy_config.json``.  Defaults to the bundled
        config resolved by :mod:`utils.config`.
    device : str, optional
        Torch device string (``"cuda"`` / ``"cpu"``).  Auto-selected when
        omitted (CUDA if available, else CPU).

    Attributes
    ----------
    modality : str
        Always ``"text"``.
    device : str
        The resolved torch device.
    """

    #: This encoder's fixed modality tag.
    modality = "text"

    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """Load RoBERTa (embeddings) and toxic-bert (6-label harm) once.

        Inputs: optional config path and device.
        Outputs: a ready-to-use ``TextEncoder`` instance with both models
        resident in memory on the chosen device.
        """
        # Import heavy deps lazily so simply importing this module (e.g. for AST
        # checks or docs) does not require torch / transformers to be installed.
        import torch
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._config_path = config_path
        self._config: Dict[str, Any] = load_policy_config(config_path)
        self._toxic_map: Dict[str, Any] = self._config.get("toxic_bert_mapping", {})

        # Auto-select device unless the caller pinned one.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        print(f"[CHEM][text_encoder] Loading models on device={self.device} ...")

        # --- RoBERTa: sentence-level semantic embedding backbone --------------
        print("[CHEM][text_encoder] Loading roberta-base for embeddings ...")
        self._emb_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self._emb_model = AutoModel.from_pretrained("roberta-base").to(self.device)
        self._emb_model.eval()

        # --- toxic-bert: 6-label toxicity classifier -------------------------
        print("[CHEM][text_encoder] Loading unitary/toxic-bert for harm labels ...")
        self._tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        self._tox_model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/toxic-bert"
        ).to(self.device)
        self._tox_model.eval()

        # Ordered list of the 6 toxic-bert label names as the model emits them.
        # ``id2label`` is the authoritative ordering shipped with the model.
        id2label = self._tox_model.config.id2label
        self._tox_labels: List[str] = [id2label[i] for i in range(len(id2label))]

        print(
            f"[CHEM][text_encoder] Ready. toxic-bert labels={self._tox_labels} "
            f"| score labels={self._toxic_map.get('text_harm_score_labels')}"
        )

    # ------------------------------------------------------------------ utils
    def _embed(self, text: str) -> List[float]:
        """Return a 768-d RoBERTa embedding for ``text``.

        Uses mean pooling over the last hidden state (attention-mask aware),
        which is a robust sentence representation for short/medium content.

        Inputs: a text string.
        Outputs: a list[float] of length 768 (via ``to_fixed_dim``).
        """
        torch = self._torch
        enc = self._emb_tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = self._emb_model(**enc)
        # Mean-pool the token embeddings using the attention mask.
        last_hidden = out.last_hidden_state  # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [1, T, 1]
        summed = (last_hidden * mask).sum(dim=1)  # [1, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [1, 1]
        pooled = (summed / counts).squeeze(0)  # [H]
        return to_fixed_dim(pooled.detach().cpu().tolist())

    def _toxic_scores(self, text: str) -> Dict[str, float]:
        """Return per-label toxic-bert probabilities for ``text``.

        toxic-bert is a multi-label classifier, so each logit is passed through
        an independent sigmoid (not a softmax).

        Inputs: a text string.
        Outputs: ``{label_name: probability}`` for all 6 labels.
        """
        torch = self._torch
        enc = self._tox_tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self._tox_model(**enc).logits.squeeze(0)  # [6]
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        return {label: float(p) for label, p in zip(self._tox_labels, probs)}

    # ------------------------------------------------------------------ encode
    def encode(self, text: str) -> Dict[str, Any]:
        """Encode ``text`` into CHEM elements plus a scalar harm score.

        Steps:
          1. Compute a 768-d RoBERTa embedding and emit a base ``text`` node.
          2. Run toxic-bert; map each of the 6 labels to a policy category via
             ``toxic_bert_mapping.labels``.
          3. For any label whose probability clears
             ``toxic_bert_mapping.harmful_signal_threshold`` emit one harmful
             policy-category node, deduplicated so a category never appears
             twice (highest-confidence occurrence wins).
          4. ``harm_score`` = max probability over
             ``toxic_bert_mapping.text_harm_score_labels``.

        Inputs
        ------
        text : str
            Raw input text.

        Outputs
        -------
        dict
            ``{"elements": list[element], "harm_score": float,
               "modality": "text"}``.
        """
        text = text or ""
        print(f"[CHEM][text_encoder] encode() len={len(text)} chars")

        base_embedding = self._embed(text)
        scores = self._toxic_scores(text)

        label_to_cat: Dict[str, str] = self._toxic_map.get("labels", {})
        signal_threshold = float(self._toxic_map.get("harmful_signal_threshold", 0.5))
        score_labels: List[str] = self._toxic_map.get("text_harm_score_labels", [])

        elements: List[Dict[str, Any]] = []
        idx = 0

        # (1) Base text node — always present, carries the semantic embedding.
        elements.append(
            {
                "id": f"text_{idx}",
                "label": "text",
                "type": "text",
                "embedding": base_embedding,
                "confidence": 1.0,
                "harmful_signal": False,
                "full_label": (text[:80] + "…") if len(text) > 80 else text,
                "modality": self.modality,
            }
        )
        idx += 1

        # (2/3) Harmful policy-category nodes, deduplicated by category id.
        # Track the best (highest-confidence) occurrence per category.
        best_per_category: Dict[str, float] = {}
        for label, prob in scores.items():
            if prob < signal_threshold:
                continue
            category = label_to_cat.get(label)
            if not category:
                continue
            if prob > best_per_category.get(category, -1.0):
                best_per_category[category] = prob

        for category, prob in sorted(
            best_per_category.items(), key=lambda kv: kv[1], reverse=True
        ):
            clause = get_policy_clause(category, self._config_path)
            elements.append(
                {
                    "id": f"text_{idx}",
                    "label": category,
                    "type": "text",
                    # Reuse the sentence embedding as the category node feature;
                    # the policy category is carried explicitly in metadata.
                    "embedding": base_embedding,
                    "confidence": float(prob),
                    "harmful_signal": True,
                    "full_label": clause.get("title", category),
                    "modality": self.modality,
                    "policy_category": category,
                }
            )
            idx += 1

        # (4) Scalar harm score = max over the configured score labels.
        harm_score = 0.0
        if score_labels:
            harm_score = max((scores.get(lbl, 0.0) for lbl in score_labels), default=0.0)

        print(
            f"[CHEM][text_encoder] -> {len(elements)} element(s), "
            f"harm_score={harm_score:.4f}, categories={list(best_per_category)}"
        )
        return {
            "elements": elements,
            "harm_score": float(harm_score),
            "modality": self.modality,
        }
