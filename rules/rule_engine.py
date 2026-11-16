"""Hand-written rule engine for text (and OCR / transcript text).

All lexicons, regex patterns, weights and context modifiers live in
``policy/rules.json``; this module only compiles and applies them.

Scoring in short
----------------
1. The text is normalised (lower-case, de-obfuscation of ``n1gg3r``-style
   spellings, repeated letters collapsed).
2. Every lexicon term and pattern that matches becomes a *match* with a base
   weight.
3. Each match is adjusted by context: negation just before it, a benign sense
   ("kill it", "got my shot"), reporting / news language, counter-speech
   ("stop asian hate"), and benign-context keywords from the policy config.
4. Matches are combined per policy category with a noisy-or; the overall score
   is the strongest category plus a small bonus per additional category.

``RuleEngine.encode`` returns the same dict shape as ``TextEncoder.encode`` so
the audio encoder can reuse it for transcripts.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from utils.config import load_policy_config
from utils.embeddings import to_fixed_dim

_DEFAULT_RULES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "policy", "rules.json")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")
#: categories that context (news, counter-speech, negation) must never excuse.
_NEVER_DAMPEN = {"child_safety", "sexual_exploitation"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return text.lower()


def _deobfuscate(text: str) -> str:
    """Undo common obfuscation: digit/symbol letter swaps and stretched letters."""
    out = []
    for tok in text.split(" "):
        if re.search(r"[a-z]", tok) and re.search(r"[01345 7@$]".replace(" ", ""), tok) and not re.fullmatch(r"[\d:.,%$/-]+\w{0,3}", tok):
            tok = tok.translate(_LEET)
        out.append(tok)
    return re.sub(r"(\w)\1{2,}", r"\1\1", " ".join(out))


class RuleEngine:
    """Compile ``rules.json`` once; score any number of texts."""

    modality = "text"

    def __init__(self, rules_path: Optional[str] = None, config_path: Optional[str] = None):
        path = rules_path or os.environ.get("CHEM_RULES_PATH") or _DEFAULT_RULES
        with open(path, "r", encoding="utf-8") as fh:
            self.rules = json.load(fh)
        self.ctx = self.rules["context"]
        self.decision = dict(self.rules["decision"])
        thr_file = os.path.join(os.path.dirname(path), "rules_threshold.json")
        if os.path.isfile(thr_file):
            try:
                self.decision["harmful_threshold"] = float(json.load(open(thr_file))["harmful_threshold"])
            except Exception:  # noqa: BLE001
                pass
        env_thr = os.environ.get("CHEM_RULES_THRESHOLD")
        if env_thr:
            self.decision["harmful_threshold"] = float(env_thr)
        self.threshold = float(self.decision["harmful_threshold"])

        cfg = load_policy_config(config_path)
        self._exculpatory = [k.lower() for k in cfg.get("exculpatory_keywords", {}).get("keywords", [])]
        self._exc_factor = 0.4

        self._compile()

    # ----------------------------------------------------------------- build
    def _expand(self, rx: str) -> str:
        for name, items in self.rules["lists"].items():
            rx = rx.replace("{" + name + "}", "|".join(items))
        return rx

    def _compile(self) -> None:
        self._rules: List[Dict[str, Any]] = []
        for i, entry in enumerate(self.rules["lexicon"]):
            body = "|".join(entry["terms"])
            rx = rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])" if entry.get("boundary", True) else rf"(?:{body})"
            self._rules.append({
                "id": f"lex_{entry['category']}_{i}", "category": entry["category"],
                "weight": float(entry["weight"]), "description": entry["description"],
                "rx": re.compile(rx, re.I),
            })
        for pat in self.rules["patterns"]:
            self._rules.append({
                "id": pat["id"], "category": pat["category"], "weight": float(pat["weight"]),
                "description": pat["description"], "rx": re.compile(self._expand(pat["regex"]), re.I),
            })
        c = self.ctx
        self._neg = re.compile(r"(?<![a-z])(?:%s)(?![a-z])" % "|".join(c["negation_before"]), re.I)
        self._report = re.compile("|".join(c["reporting_markers"]), re.I)
        self._counter = re.compile("|".join(c["counter_speech_markers"]), re.I)
        self._joke = re.compile("|".join(c["joke_markers"]), re.I)
        self._benign = [(re.compile(f"^(?:{k})$", re.I), re.compile(r"(?:%s)" % "|".join(v), re.I)) for k, v in c["benign_after"].items()]
        self._benign_after = [(k, re.compile(r"^\W*(?:the |a |an |my |your |his |her |our |their |some |that |this )?(?:%s)" % "|".join(v), re.I)) for k, v in c["benign_after"].items()]
        self._benign_before = [(k, re.compile(r"(?:%s)\W*$" % "|".join(v), re.I)) for k, v in c["benign_after"].items()]
        self._benign_keys = [(re.compile(k, re.I), a, b) for (k, a), (_, b) in zip(self._benign_after, self._benign_before)]

    # ---------------------------------------------------------------- context
    def _is_benign(self, text: str, start: int, end: int) -> bool:
        matched = text[start:end]
        after = text[end:end + int(self.ctx["benign_window_chars"])]
        before = text[max(0, start - 20):start]
        for key_rx, after_rx, before_rx in self._benign_keys:
            if key_rx.search(matched) and (after_rx.search(after) or before_rx.search(before)):
                return True
        return False

    def _negated(self, text: str, start: int) -> bool:
        sent_start = max(text.rfind(ch, 0, start) for ch in ".!?;\n") + 1
        window = text[sent_start:start].split()[-int(self.ctx["negation_window_tokens"]):]
        return bool(self._neg.search(" ".join(window)))

    # ---------------------------------------------------------------- analyse
    def analyze(self, text: Optional[str]) -> Dict[str, Any]:
        """Score ``text``. Returns score, per-category scores and every match."""
        raw = text or ""
        base = _norm(raw)
        empty = {"score": 0.0, "category_scores": {}, "matches": [], "context": []}
        if len(base.strip()) < int(self.decision.get("min_text_chars_for_context", 3)):
            return empty
        variants = [base]
        alt = _deobfuscate(base)
        if alt != base:
            variants.append(alt)

        doc_flags: List[str] = []
        reporting = bool(self._report.search(base))
        counter = bool(self._counter.search(base))
        joking = bool(self._joke.search(base))
        exculpatory = any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", base) for k in self._exculpatory)
        if reporting:
            doc_flags.append("news/reporting language")
        if counter:
            doc_flags.append("counter-speech language")
        if exculpatory:
            doc_flags.append("benign-context keyword")

        found: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for vtext in variants:
            for rule in self._rules:
                for m in rule["rx"].finditer(vtext):
                    s, e = m.span()
                    snippet = vtext[s:e].strip()
                    if not snippet:
                        continue
                    mods: List[str] = []
                    mult = 1.0
                    cat = rule["category"]
                    if self._negated(vtext, s):
                        mult *= float(self.ctx["negation_multiplier"]); mods.append("negated")
                    if self._is_benign(vtext, s, e):
                        mult *= float(self.ctx["benign_multiplier"]); mods.append("benign sense")
                    if cat not in _NEVER_DAMPEN:
                        if reporting:
                            mult *= float(self.ctx["reporting_multiplier"]); mods.append("reporting context")
                        if counter and cat in self.ctx["counter_speech_applies_to"]:
                            mult *= float(self.ctx["counter_speech_multiplier"]); mods.append("counter-speech")
                        if exculpatory and cat in ("weapons_threat", "violent_threat", "graphic_violence"):
                            mult *= self._exc_factor; mods.append("benign-context keyword")
                    if joking:
                        mult *= float(self.ctx["joke_multiplier"]); mods.append("joking tone")
                    key = (rule["id"], snippet)
                    item = {"rule_id": rule["id"], "category": cat, "weight": rule["weight"],
                            "adjusted": round(rule["weight"] * mult, 4), "text": snippet,
                            "description": rule["description"], "modifiers": mods}
                    if key not in found or item["adjusted"] > found[key]["adjusted"]:
                        found[key] = item

        matches = list(found.values())
        # weapon / violence emoji
        emo = [e for e in self.ctx["emoji_weapon"] if e in raw]
        if emo:
            matches.append({"rule_id": "emoji_weapon", "category": "weapons_threat", "weight": self.ctx["emoji_weapon_weight"],
                            "adjusted": float(self.ctx["emoji_weapon_weight"]), "text": "".join(emo[:3]),
                            "description": "weapon / violence emoji", "modifiers": []})
        # collapse overlapping matches inside the same category: keep the strongest per span text containment
        matches = self._dedupe(matches)

        cat_scores: Dict[str, float] = {}
        for cat in {m["category"] for m in matches}:
            prod = 1.0
            for m in matches:
                if m["category"] == cat:
                    prod *= 1.0 - min(0.99, m["adjusted"])
            cat_scores[cat] = round(1.0 - prod, 4)

        score = 0.0
        if cat_scores:
            top = max(cat_scores.values())
            extra = sum(1 for v in cat_scores.values() if v >= 0.3) - 1
            score = top + float(self.decision["diversity_bonus"]) * max(0, min(2, extra))
            if top >= 0.3:
                letters = [ch for ch in raw if ch.isalpha()]
                if len(letters) >= self.ctx["all_caps_min_letters"] and sum(ch.isupper() for ch in letters) / len(letters) >= self.ctx["all_caps_ratio"]:
                    score += float(self.ctx["all_caps_bonus"])
                score += min(float(self.ctx["exclaim_bonus_max"]), raw.count("!") * float(self.ctx["exclaim_bonus_per"]))
            score = min(float(self.decision["max_score"]), score)
        return {"score": round(score, 4), "category_scores": cat_scores,
                "matches": sorted(matches, key=lambda m: -m["adjusted"]), "context": doc_flags}

    @staticmethod
    def _dedupe(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop a match whose text is wholly contained in a stronger same-category match."""
        keep: List[Dict[str, Any]] = []
        for m in sorted(matches, key=lambda x: (-x["adjusted"], -len(x["text"]))):
            if any(k["category"] == m["category"] and m["text"] in k["text"] and k is not m for k in keep):
                continue
            keep.append(m)
        return keep

    # --------------------------------------------------- TextEncoder interface
    def encode(self, text: str) -> Dict[str, Any]:
        """Return a ``TextEncoder``-compatible dict (used for audio transcripts)."""
        res = self.analyze(text)
        elements: List[Dict[str, Any]] = [{
            "id": "text_0", "label": "text", "type": "text", "embedding": to_fixed_dim([0.0]),
            "confidence": 1.0, "harmful_signal": False, "full_label": (text or "")[:80], "modality": "text",
        }]
        for i, m in enumerate(res["matches"], start=1):
            if m["adjusted"] < 0.3:
                continue
            elements.append({
                "id": f"text_{i}", "label": m["category"], "type": "text", "embedding": to_fixed_dim([0.0]),
                "confidence": m["adjusted"], "harmful_signal": True, "policy_category": m["category"],
                "full_label": f"{m['description']}: \"{m['text']}\"", "modality": "text",
            })
        return {"elements": elements, "harm_score": res["score"], "modality": "text", "analysis": res}
