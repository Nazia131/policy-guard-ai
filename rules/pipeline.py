"""Multimodal pipeline for the rule-based prototype.

Text, OCR text from images, CLIP concept scores for images (optional) and audio
transcripts all feed the same :class:`rules.rule_engine.RuleEngine`.  The
response has the same shape as the model-based pipeline so the frontend is
unchanged.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from rules.rule_engine import RuleEngine
from utils.config import get_policy_clause


class RulesPipeline:
    """Rule-based analysis of text / image / audio."""

    def __init__(self, use_clip: bool = True, use_audio: bool = True) -> None:
        self.engine = RuleEngine()
        self.cfg_image = self.engine.rules["image"]
        self.cfg_audio = self.engine.rules["audio"]
        self.image_encoder = None
        self.audio_encoder = None
        self.notes: List[str] = []
        if use_clip:
            try:
                from encoders.image_encoder import ImageEncoder

                self.image_encoder = ImageEncoder()
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"image model unavailable ({type(exc).__name__}); image analysis uses OCR text only")
                print(f"[CHEM][rules] CLIP unavailable: {exc}")
        self.probe = None
        if self.image_encoder is not None:
            try:
                import os
                from model.clip_linear import DEFAULT_PATH, ClipLinear

                if os.path.isfile(os.environ.get("CHEM_CLIP_LINEAR_PATH") or DEFAULT_PATH):
                    self.probe = ClipLinear()
                    print("[CHEM][rules] CLIP image+text probe loaded.")
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"trained probe unavailable ({type(exc).__name__})")
        if use_audio:
            try:
                from encoders.audio_encoder import AudioEncoder

                self.audio_encoder = AudioEncoder(text_encoder=self.engine, whisper_model=self.cfg_audio.get("whisper_model", "base"))
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"audio unavailable ({type(exc).__name__})")
                print(f"[CHEM][rules] Audio unavailable: {exc}")

    # ------------------------------------------------------------------ image
    def _image_scores(self, pil, text_score: float) -> Dict[str, Any]:
        """CLIP concept hits -> hand-tuned image matches (empty if CLIP is absent)."""
        matches: List[Dict[str, Any]] = []
        if self.image_encoder is None:
            return {"matches": matches}
        res = self.image_encoder.encode(pil)
        cfg = self.cfg_image
        for el in res.get("elements", []):
            cat = el.get("policy_category")
            if not cat or el.get("label") == "image" and not el.get("harmful_signal"):
                continue
            conf = float(el.get("confidence", 0.0))
            if conf < float(cfg["min_clip_score_to_report"]):
                continue
            adj = conf * float(cfg["clip_weight"]) * float(cfg["category_scale"].get(cat, 1.0))
            mods: List[str] = []
            if text_score <= float(cfg["benign_text_max_text_score"]) and text_score > 0:
                adj *= 1.0 - float(cfg["benign_text_dampening"]); mods.append("benign caption")
            matches.append({"rule_id": "clip_" + cat, "category": cat, "weight": conf, "adjusted": round(min(adj, 0.99), 4),
                            "text": el.get("full_label", cat), "description": "image concept match", "modifiers": mods})
        return {"matches": matches}

    # ---------------------------------------------------------------- analyse
    def analyze(self, text: Optional[str] = None, image_bytes: Optional[bytes] = None,
                audio_path: Optional[str] = None, video_path: Optional[str] = None) -> Dict[str, Any]:
        eng = self.engine
        mod_matches: Dict[str, List[Dict[str, Any]]] = {}
        mod_scores: Dict[str, float] = {}
        mod_context: Dict[str, List[str]] = {}
        transcript = ""
        image_text = ""
        exculpatory_text = text or ""

        if text and text.strip():
            r = eng.analyze(text)
            mod_matches["text"], mod_scores["text"], mod_context["text"] = r["matches"], r["score"], r["context"]

        if image_bytes is not None:
            from PIL import Image

            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            try:
                from encoders.ocr import extract_text

                image_text = extract_text(pil)
            except Exception as exc:  # noqa: BLE001
                print(f"[CHEM][rules] OCR skipped: {exc}")
            ocr = eng.analyze(image_text) if image_text else {"matches": [], "score": 0.0, "context": []}
            clip = self._image_scores(pil, mod_scores.get("text", 0.0))
            matches = [dict(m, text=f"{m['text']} (read from image)") for m in ocr["matches"]]
            matches += clip["matches"]
            score = 0.0
            if matches:
                prod = 1.0
                for m in matches:
                    prod *= 1.0 - min(0.99, m["adjusted"])
                score = max(ocr["score"], 1.0 - prod)
            mod_matches["image"], mod_scores["image"], mod_context["image"] = matches, round(min(score, 0.99), 4), ocr["context"]

        probe_p = None
        if image_bytes is not None and self.probe is not None:
            try:
                from model.clip_linear import clip_embed

                ie = self.image_encoder
                fi, ft = clip_embed(ie._model, ie._preprocess, ie.device, [pil], [(text or "").strip() or image_text])
                probe_p = float(self.probe.predict_proba(fi, ft)[0])
                mod_scores["multimodal"] = round(max(0.0, min(0.99, self.probe.to_score(probe_p))), 4)
            except Exception as exc:  # noqa: BLE001
                print(f"[CHEM][rules] probe failed: {exc}")

        if audio_path is not None:
            if self.audio_encoder is None:
                raise ValueError("Audio analysis is unavailable on this machine (needs: pip install openai-whisper librosa soundfile).")
            res = self.audio_encoder.encode(audio_path)
            transcript = res.get("transcript", "")
            analysis = (res.get("analysis") or {})
            a = eng.analyze(transcript) if transcript else {"matches": [], "score": 0.0, "context": []}
            mod_matches["audio"], mod_context["audio"] = a["matches"], a["context"]
            mod_scores["audio"] = float(res.get("harm_score", 0.0))

        if not mod_scores and not image_bytes:
            return self._empty()

        # exculpatory text dampens image-only matches
        dec = eng.decision
        base = max(mod_scores.values()) if mod_scores else 0.0
        strong = [m for m, s in mod_scores.items() if s >= float(dec["cross_modal_min_each"])]
        cross = len(strong) >= 2
        final = min(float(dec["max_score"]), base + (float(dec["cross_modal_bonus"]) if cross else 0.0))
        harmful = final >= eng.threshold

        all_matches = [(mod, m) for mod, ms in mod_matches.items() for m in ms]
        all_matches.sort(key=lambda t: -t[1]["adjusted"])
        top_cat = None
        if all_matches and harmful:
            cat_best: Dict[str, float] = {}
            for _, m in all_matches:
                cat_best[m["category"]] = max(cat_best.get(m["category"], 0.0), m["adjusted"])
            top_cat = max(cat_best, key=cat_best.get)
        clause = get_policy_clause(top_cat) if top_cat else None

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        triggered: List[Dict[str, Any]] = []
        for mod in mod_scores:
            nodes.append({"id": mod, "label": mod, "type": mod, "modality": mod,
                          "harmful_signal": False, "confidence": mod_scores[mod], "full_label": f"{mod} content"})
        if cross and len(strong) >= 2:
            edges.append({"source_id": strong[0], "target_id": strong[1], "edge_type": 2})
        for i, (mod, m) in enumerate(all_matches):
            if m["adjusted"] < 0.15:
                continue
            flag = m["adjusted"] >= 0.3
            node = {"id": f"{mod}_m{i}", "label": m["category"], "type": mod, "modality": mod, "harmful_signal": flag,
                    "confidence": m["adjusted"], "policy_category": m["category"],
                    "full_label": f"{m['description']}: \"{m['text']}\"", "text": m["text"]}
            nodes.append(node)
            edges.append({"source_id": mod, "target_id": node["id"], "edge_type": 4 if flag else 0})
            if flag and harmful:
                triggered.append(node)

        explanation = self._explain(harmful, final, clause, all_matches, mod_context, cross, image_text, transcript)
        if probe_p is not None:
            explanation += (f" Trained image+text classifier (CLIP + logistic regression) gave a harm probability of {probe_p:.2f}"
                            f" (its decision threshold is {self.probe.threshold:.2f}); it does not name a policy category.")
        return {
            "engine": "rules", "harm_score": round(final, 4), "verdict": "HARMFUL" if harmful else "SAFE",
            "policy_clause": clause if harmful else None,
            "triggered_category": clause.get("clause_id") if (harmful and isinstance(clause, dict)) else None,
            "triggered_elements": triggered, "nodes": nodes, "edges": edges, "explanation": explanation,
            "cross_modal_flag": cross,
            "shap_scores": [{"element": f"{mod}:{m['text']}", "score": m["adjusted"]} for mod, m in all_matches[:12]],
            "modality_scores": {k: round(v, 4) for k, v in mod_scores.items()},
            "image_text": image_text, "transcript": transcript, "notes": self.notes,
        }

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"engine": "rules", "harm_score": 0.0, "verdict": "SAFE", "policy_clause": None, "triggered_category": None,
                "triggered_elements": [], "nodes": [], "edges": [], "explanation": "No content was provided to analyse.",
                "cross_modal_flag": False, "shap_scores": [], "modality_scores": {}, "image_text": "", "transcript": "", "notes": []}

    def _explain(self, harmful, score, clause, matches, context, cross, image_text, transcript) -> str:
        parts: List[str] = []
        shown = [(mod, m) for mod, m in matches if m["adjusted"] >= (0.3 if harmful else 0.15)][:4]
        if harmful and isinstance(clause, dict):
            parts.append(f"Flagged under {clause.get('title')} ({clause.get('clause_id')}): {clause.get('text')}")
            if shown:
                bits = []
                for mod, m in shown:
                    src = {"text": "text", "image": "image", "audio": "audio"}.get(mod, mod)
                    bits.append(f"\"{m['text']}\" in the {src} ({m['description']}, {m['adjusted']:.2f})")
                parts.append("Triggered by: " + "; ".join(bits) + ".")
            if cross:
                parts.append("More than one modality carried a harmful signal, which raised the score.")
        else:
            if shown:
                bits = [f"\"{m['text']}\" ({m['description']}, {m['adjusted']:.2f})" for _, m in shown]
                parts.append("No policy violation was detected. Weak signals seen: " + "; ".join(bits) + ".")
            else:
                parts.append("No policy violation was detected. No harmful language or imagery matched the rule set.")
            parts.append(f"Combined score {score:.2f} is below the {self.engine.threshold:.2f} decision threshold.")
        mods = sorted({x for _, m in shown for x in m.get("modifiers", [])})
        flags = sorted({f for fl in context.values() for f in fl})
        notes = sorted(set(mods) | set(flags))
        if notes:
            parts.append("Context that lowered the score: " + ", ".join(notes) + ".")
        if image_text:
            parts.append(f"Text read from the image: \"{image_text[:160]}\".")
        if transcript:
            parts.append(f"Audio transcript: \"{transcript[:160]}\".")
        parts.append("(Rule-based prototype: hand-written rules, not a trained model.)")
        return " ".join(parts)
