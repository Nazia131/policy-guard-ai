# CHEM — Compositional Harm Emergence Modeling

**A Policy-Aware Multimodal AI Content-Moderation System with Explainable Decision Intelligence.**

CHEM detects harm that *emerges from the composition* of otherwise-innocuous
elements across text, image, audio and video. Instead of scoring each modality
in isolation, CHEM decomposes content into typed **elements**, wires them into a
**Content Composition Graph (CCG)**, and classifies the graph with a **Graph
Attention Network (GAT)**. Decisions are grounded in a **Harm Pattern Library**
mined from data and mapped to real platform policy clauses (Meta Community
Standards, YouTube Community Guidelines, EU DSA), and every decision is
**explained** with SHAP attributions and a GNNExplainer minimal subgraph.

> **Scope & intent.** CHEM is a defensive content-moderation research system for
> a final-year engineering project. It is designed to *detect and explain*
> harmful content, not to produce it.

---

## Key properties

- **Nothing is hardcoded.** Every harm category, visual concept, policy clause
  and numeric coefficient lives in [`policy/policy_config.json`](policy/policy_config.json).
  Python modules read from it via `utils/config.py`.
- **Compositional.** Harm is modelled as a *pattern of relationships* between
  elements, not a single toxic token or pixel.
- **Cross-modal fusion is dynamic.** Modality weights normalise over whichever
  modalities are actually present; a cross-modal bonus is added only when
  harmful signals co-occur across ≥2 modalities (with per-modality minimums).
- **Explainable by construction.** SHAP + GNNExplainer + matched library pattern
  → a human-readable rationale citing the exact policy clause.

---

## Architecture

```
             ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  text ────▶ │ TextEncoder │  │ ImageEncoder │  │ AudioEncoder │  │ VideoEncoder │
  image ───▶ │  RoBERTa +  │  │  CLIP ViT-   │  │  Whisper +   │  │  VideoMAE +  │
  audio ───▶ │  toxic-bert │  │  B/32 (0-shot│  │  librosa +   │  │  CLIP frames │
  video ───▶ │             │  │  vs concepts)│  │  toxic-bert  │  │              │
             └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                    └──────── typed "elements" (768-d embeddings) ───────┘
                                          │
                              ┌───────────▼────────────┐
                              │  ContentCompositionGraph│  co-occurrence / semantic /
                              │  Builder  → PyG Data     │  temporal / target-pointing /
                              └───────────┬────────────┘  harmful-pattern edges
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │  GAT (2× GATConv) + global mean pool + head │
                    └───────────────┬─────────────┬──────────────┘
                                    │             │
                    ┌───────────────▼──┐   ┌──────▼───────────────┐
                    │  Harm Pattern     │   │  Explainability      │
                    │  Library (graph   │   │  SHAP + GNNExplainer │
                    │  similarity match)│   │  → policy rationale  │
                    └───────────────────┘   └──────────────────────┘
                                    │
                              ┌─────▼──────┐
                              │ FastAPI    │  POST /analyze  ·  GET /health
                              └────────────┘
```

---

## Repository layout

```
chem-final/
├── data/                     # dataset download + preprocessing
│   ├── download_datasets.py
│   └── preprocess/{hateful_memes.py, ucf_crime.py}
├── encoders/                 # per-modality decomposition into typed elements
│   ├── text_encoder.py       # RoBERTa + unitary/toxic-bert
│   ├── image_encoder.py      # CLIP ViT-B/32 zero-shot vs config concepts
│   ├── audio_encoder.py      # Whisper transcript + librosa acoustics
│   └── video_encoder.py      # VideoMAE actions + CLIP frames
├── graph/                    # Content Composition Graph
│   ├── edge_types.py
│   ├── graph_builder.py      # → torch_geometric.data.Data + JSON node/edge list
│   └── visualize_graph.py
├── harm_pattern_library/     # mine → tag → build library.json
│   ├── subgraph_miner.py     # frequent subgraph mining (gSpan-style / NetworkX)
│   ├── pattern_tagger.py     # rule-based → policy clause
│   └── library_builder.py
├── model/
│   ├── gat_model.py          # CHEMGraphAttentionNetwork
│   ├── harm_scorer.py        # inference + graph-similarity + dynamic fusion
│   └── train.py              # Hateful Memes training, early stopping, W&B
├── explainability/
│   ├── shap_explainer.py     # SHAP GradientExplainer over the GAT
│   ├── gnn_explainer.py      # PyG GNNExplainer minimal subgraph
│   └── report_generator.py   # structured + human-readable rationale
├── policy/policy_config.json # SINGLE SOURCE OF TRUTH
├── evaluate/{metrics.py, zero_shot_eval.py, ablation_study.py}
├── api/app.py                # FastAPI service
├── utils/{config.py, embeddings.py}
├── requirements.txt
└── README.md
```

---

## 1. Setup

```bash
# Python 3.10+ recommended. Use a fresh virtual environment.
python3 -m venv .venv && source .venv/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

# torch-geometric wheels must match your installed torch/CUDA build.
# If the pip line in requirements.txt fails, install it explicitly, e.g.:
#   pip install torch-geometric
# See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

Model weights (RoBERTa, toxic-bert, CLIP, Whisper, VideoMAE) download
automatically from Hugging Face / OpenAI on first use and are cached locally.
A GPU is strongly recommended for training; inference runs on CPU.

Always run commands **from the project root** so absolute imports resolve:

```bash
cd chem-final
export PYTHONPATH="$PWD:$PYTHONPATH"   # or run modules with `python -m`
```

---

## 2. Download datasets

### Facebook Hateful Memes (primary training set)

```bash
python3 -m data.download_datasets --dataset hateful_memes --target-dir data/hateful_memes
```

The Hateful Memes dataset is released under terms that require accepting a
license. If automated download is unavailable, the script prints exact manual
instructions (via [facebookresearch/mmf](https://github.com/facebookresearch/mmf)
and the DrivenData competition page). Expected extracted layout:

```
data/hateful_memes/
├── img/                 # *.png meme images
├── train.jsonl          # {id, img, text, label}
├── dev.jsonl
└── test.jsonl
```

### UCF-Crime (video anomaly evaluation)

UCF-Crime is hosted on Kaggle and must be downloaded manually:

1. Create a free Kaggle account and accept the dataset terms.
2. Search **"UCF Crime Dataset"** on <https://www.kaggle.com/datasets>.
3. Download and extract into `data/ucf_crime/` with per-category subfolders
   (e.g. `Abuse/`, `Assault/`, … and `Normal_Videos/`).

```bash
python3 -m data.download_datasets --dataset ucf_crime   # prints the instructions
```

---

## 3. Build the Harm Pattern Library

Mines frequent harmful subgraphs from the training set and tags each with a
policy clause:

```bash
python3 -m harm_pattern_library.library_builder \
    --data-dir data/hateful_memes \
    --min-support 0.05 \
    --out harm_pattern_library/library.json
```

---

## 4. Train the GAT

```bash
python3 -m model.train \
    --data-dir data/hateful_memes \
    --epochs 200 \
    --batch-size 32 \
    --wandb-project chem-final
```

- 70/15/15 train/val/test split, Adam (`lr=5e-3`, `weight_decay=5e-4`),
  early stopping (patience 20) on validation **F1**.
- Best checkpoint saved to `model/checkpoints/best_gat.pt`.
- Metrics logged to Weights & Biases (set `WANDB_MODE=offline` to run without an
  account). Final test accuracy / precision / recall / F1 / AUC are printed.

---

## 5. Run the API

```bash
# Optional: point at your trained artifacts (defaults shown).
export CHEM_MODEL_PATH=model/checkpoints/best_gat.pt
export CHEM_LIBRARY_PATH=harm_pattern_library/library.json

uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Analyse content (any subset of modalities):

```bash
curl -X POST http://localhost:8000/analyze \
  -F 'text=they should not be allowed here' \
  -F 'image=@examples/meme.png'
```

Response (abridged):

```json
{
  "harm_score": 0.71,
  "verdict": "HARMFUL",
  "policy_clause": {"clause_id": "MCS-3.2", "title": "Hate Speech", "source": "Meta Community Standards"},
  "triggered_category": "MCS-3.2",
  "triggered_elements": [...],
  "nodes": [...],
  "edges": [...],
  "explanation": "This content was flagged under Hate Speech. The following elements combined to form a harmful pattern: ...",
  "cross_modal_flag": true,
  "shap_scores": [...],
  "modality_scores": {"text": 0.62, "image": 0.44}
}
```

---

## 6. Evaluation

```bash
# Standard metrics on the held-out test set (accuracy / precision / recall / F1 / AUC + per-class)
python3 -m evaluate.metrics --checkpoint model/checkpoints/best_gat.pt --data-dir data/hateful_memes

# Zero-shot structural generalisation: hold out one entire harm category
python3 -m evaluate.zero_shot_eval --held-out-category weapons_threat --data-dir data/hateful_memes

# Ablations: no-graph / no-policy-tags / no-temporal-edges / GCN-instead-of-GAT
python3 -m evaluate.ablation_study --data-dir data/hateful_memes
```

---

## Design guarantees (project constraints)

| Constraint | Where it is enforced |
|---|---|
| No hardcoded harm categories / concepts / clauses | `policy/policy_config.json` + `utils/config.py` |
| No hardcoded acoustic thresholds | `encoders/audio_encoder.py` — agitation is `pitch_std/pitch_mean` relative to the signal |
| Dynamic modality weights | `model/harm_scorer.fuse_modality_scores` normalises over present modalities |
| Cross-modal bonus (0.2), min image 0.3 / audio 0.25 | `coefficients` in config, applied in fusion |
| Models loaded once | encoder/scorer classes load in `__init__`; API loads at startup |
| Every function/class documented | docstrings throughout |

---

## Testing

```bash
pytest -q
```

Unit tests focus on the pure-Python components that do not require the heavy
model stack (config loading, embedding normalisation, graph construction logic,
pattern tagging rules, metric computation).

---

## Citation

If you use CHEM in academic work, please cite this repository and the underlying
datasets (Hateful Memes; UCF-Crime) and models (RoBERTa, CLIP, Whisper,
VideoMAE, toxic-bert) per their respective licenses.
