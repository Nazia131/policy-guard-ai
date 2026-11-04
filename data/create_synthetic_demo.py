"""Synthetic demo-data generator for CHEM pipeline testing.

Creates a small Hateful Memes-format dataset (JSONL + placeholder images) under
``data/hateful_memes/hateful_memes/`` so the full CHEM pipeline can be exercised
end-to-end *without* registering for the gated DrivenData release.

Usage::

    python3 -m data.create_synthetic_demo [--n-samples 60] [--out-dir data/hateful_memes/hateful_memes]

Generates:
    img/<id>.png           -- small solid-colour images with PIL
    train.jsonl            -- 70 % of samples
    dev.jsonl              -- 15 % of samples
    test.jsonl             -- 15 % of samples

Each JSONL line matches the real Hateful Memes schema::

    {"id": int, "img": "img/<id>.png", "text": str, "label": 0|1}

Harmful samples contain text drawn from policy-neutral paraphrase templates that
carry surface signals the text encoder can pick up.  Safe samples contain benign
captions.  The images are solid colour blocks — sufficient for CLIP zero-shot but
carrying no real visual harm signal.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from utils.provenance import write_synthetic_marker

# ---------------------------------------------------------------------------
# Sample templates: policy-neutral paraphrases (NOT real hateful content).
# These are deliberately sanitised to be safe while still exercising the
# toxic-bert / text encoder harm detection path.
# ---------------------------------------------------------------------------

_HARMFUL_TEXTS = [
    "we don't want your kind here",
    "they should all be deported",
    "these people are vermin",
    "nobody trusts people like them",
    "they are inferior and dangerous",
    "get out of our country forever",
    "people like that should be eliminated",
    "their kind doesn't belong anywhere",
    "we must stop them before it's too late",
    "they are destroying everything we built",
    "their religion is a disease",
    "they are criminals by nature",
    "no place for them in our society",
    "they are invaders we must resist",
    "their blood is poisoning our land",
    "these outsiders should go back",
    "they are taking what is ours",
    "the enemy within must be removed",
    "we need to cleanse our community",
    "they spread lies and corruption",
]

_SAFE_TEXTS = [
    "a sunny day at the park",
    "enjoying a slice of pizza with friends",
    "the mountains were beautiful this morning",
    "my cat knocked over the plant again",
    "graduation day was unforgettable",
    "finally finished reading that novel",
    "spring flowers are blooming in the garden",
    "homemade soup is the best on cold days",
    "the team worked hard to finish the project",
    "the kids had a great time at the beach",
    "baking cookies with the family",
    "took a long walk by the river",
    "watched a really funny movie last night",
    "the new library opened downtown today",
    "morning coffee makes everything better",
    "the museum exhibit was fascinating",
    "teaching my nephew to ride a bicycle",
    "the concert was absolutely amazing",
    "planted tomatoes in the backyard today",
    "the weekend hike recharged my energy",
]

_COLOURS_SAFE = [
    (135, 206, 250),   # sky blue
    (144, 238, 144),   # light green
    (255, 255, 153),   # light yellow
    (255, 218, 185),   # peach
    (221, 160, 221),   # plum light
]

_COLOURS_HARMFUL = [
    (200, 80,  80),    # muted red
    (120, 60,  60),    # dark red
    (180, 100, 60),    # brown-red
    (160, 80,  40),    # dark orange
    (100, 50,  50),    # very dark red
]


def _make_image(path: str, colour: tuple) -> None:
    """Save a solid-colour 224×224 PNG to *path*.

    Parameters
    ----------
    path : str
        Absolute or relative path for the output PNG.
    colour : tuple of int
        ``(R, G, B)`` fill colour.
    """
    try:
        from PIL import Image

        img = Image.new("RGB", (224, 224), colour)
        img.save(path)
    except ImportError:
        # PIL not yet installed: write a 1-byte placeholder so the path exists
        # and the loader can at least open something.
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)


def _write_jsonl(path: str, records: list) -> None:
    """Write a list of dicts as newline-delimited JSON to *path*.

    Parameters
    ----------
    path : str
        Output file path.
    records : list of dict
        Records to serialise.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[CHEM][synthetic] Wrote {len(records)} records -> {path}")


def create_synthetic_demo(n_samples: int = 60, out_dir: str = "data/hateful_memes/hateful_memes") -> None:
    """Generate synthetic Hateful Memes-format data for pipeline testing.

    Parameters
    ----------
    n_samples : int
        Total number of samples to create.  At least 20 recommended so that
        the 70/15/15 train/val/test split has enough samples per split.
    out_dir : str
        Target root directory.  Images go into ``<out_dir>/img/`` and splits
        into ``<out_dir>/train.jsonl`` etc.
    """
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    print(f"[CHEM][synthetic] Generating {n_samples} synthetic samples in {out_dir}")

    rng = random.Random(42)
    records = []

    for i in range(n_samples):
        label = i % 2  # balanced: alternating harmful / safe
        sample_id = 10000 + i
        img_rel = f"img/{sample_id}.png"
        img_abs = os.path.join(out_dir, img_rel)

        if label == 1:
            text = rng.choice(_HARMFUL_TEXTS)
            colour = rng.choice(_COLOURS_HARMFUL)
        else:
            text = rng.choice(_SAFE_TEXTS)
            colour = rng.choice(_COLOURS_SAFE)

        _make_image(img_abs, colour)
        records.append({"id": sample_id, "img": img_rel, "text": text, "label": label})

    rng.shuffle(records)

    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    train = records[:n_train]
    dev = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev)
    _write_jsonl(os.path.join(out_dir, "test.jsonl"), test)

    # Marker so model/train.py can label any checkpoint trained on this
    # directory as synthetic-only (it must never pass as a trained model).
    marker = write_synthetic_marker(out_dir)
    print(f"[CHEM][synthetic] Wrote synthetic-data marker -> {marker}")

    print(f"[CHEM][synthetic] Done.  train={len(train)}, dev={len(dev)}, test={len(test)}")
    print(f"[CHEM][synthetic] Layout ready at: {os.path.abspath(out_dir)}")


def main() -> None:
    """Parse CLI arguments and call :func:`create_synthetic_demo`."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Hateful Memes-format demo data for CHEM."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=60,
        help="Total samples to generate (default: 60).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/hateful_memes/hateful_memes",
        help="Output directory root (default: data/hateful_memes/hateful_memes).",
    )
    args = parser.parse_args()
    create_synthetic_demo(n_samples=args.n_samples, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
