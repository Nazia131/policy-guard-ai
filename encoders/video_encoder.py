"""CHEM video encoder.

Combines two temporal views of a video:

* **Per-frame appearance** -- frames are sampled at regular intervals and each
  is passed through a reused :class:`ImageEncoder` (CLIP zero-shot), yielding
  visual elements tagged with the time (seconds) at which the frame occurred.
* **Temporal action** -- a VideoMAE action classifier
  (``MCG-NJU/videomae-base``, Kinetics-400) predicts the dominant action across
  the sampled clip; the predicted label is mapped to a policy category via
  ``video_action_mapping.harmful_actions`` (case-insensitive substring match).

Every emitted element carries a ``temporal_position``.  The clip harm score is
the maximum harmful signal observed across frames and the action prediction.

Public API
----------
``VideoEncoder(image_encoder=None, config_path=None, device=None)``
``VideoEncoder.encode(video_path, num_frames=16) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from utils.config import get_policy_clause, load_policy_config
from utils.embeddings import to_fixed_dim
from encoders.image_encoder import ImageEncoder


class VideoEncoder:
    """Encode video into CHEM elements via per-frame CLIP + VideoMAE actions.

    VideoMAE and a reusable :class:`ImageEncoder` are loaded once in
    :meth:`__init__`.

    Parameters
    ----------
    image_encoder : ImageEncoder, optional
        Pre-built image encoder to reuse for per-frame CLIP; a new one is
        constructed if not supplied.
    config_path : str, optional
        Override path to ``policy_config.json``.
    device : str, optional
        Torch device string; auto-selected when omitted.

    Attributes
    ----------
    modality : str
        Always ``"video"``.
    """

    #: This encoder's fixed modality tag.
    modality = "video"

    def __init__(
        self,
        image_encoder: Optional[ImageEncoder] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Load VideoMAE once and attach/create a reusable ImageEncoder.

        Inputs: optional image encoder, config path, device.
        Outputs: a ready ``VideoEncoder`` instance.
        """
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self._config_path = config_path
        self._config = load_policy_config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        # Reuse or build the per-frame CLIP image encoder.
        if image_encoder is not None:
            print("[CHEM][video_encoder] Reusing provided ImageEncoder.")
            self._image_encoder = image_encoder
        else:
            print("[CHEM][video_encoder] No ImageEncoder provided; building one.")
            self._image_encoder = ImageEncoder(config_path=config_path, device=device)

        print(
            f"[CHEM][video_encoder] Loading VideoMAE MCG-NJU/videomae-base "
            f"on device={self.device} ..."
        )
        self._processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self._model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base"
        ).to(self.device)
        self._model.eval()
        self._id2label = self._model.config.id2label

        # Harmful action substring -> policy category map (case-insensitive).
        self._harmful_actions: Dict[str, str] = (
            self._config.get("video_action_mapping", {}).get("harmful_actions", {})
        )

        print(
            f"[CHEM][video_encoder] Ready. {len(self._harmful_actions)} harmful "
            f"action mapping(s), {len(self._id2label)} VideoMAE classes."
        )

    # ------------------------------------------------------------------ frames
    def _read_frames(self, video_path: str, num_frames: int) -> Tuple[List[Any], List[float]]:
        """Sample ``num_frames`` frames at regular intervals from a video.

        Tries decord first, then OpenCV (cv2), then torchvision.io — whichever
        is importable — so the encoder degrades gracefully by environment.

        Inputs: video path and the number of frames to sample.
        Outputs: ``(frames, timestamps)`` where ``frames`` are RGB PIL.Images
        and ``timestamps`` are the frame times in seconds.
        """
        # --- Attempt 1: decord ----------------------------------------------
        try:
            import decord  # type: ignore
            from PIL import Image

            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            total = len(vr)
            fps = float(vr.get_avg_fps()) or 25.0
            if total == 0:
                return [], []
            indices = self._even_indices(total, num_frames)
            batch = vr.get_batch(indices).asnumpy()  # [N, H, W, 3]
            frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(len(indices))]
            timestamps = [idx / fps for idx in indices]
            print(f"[CHEM][video_encoder] Read {len(frames)} frames via decord.")
            return frames, timestamps
        except Exception as exc:
            print(f"[CHEM][video_encoder] decord unavailable/failed ({exc}); trying cv2.")

        # --- Attempt 2: OpenCV ----------------------------------------------
        try:
            import cv2  # type: ignore
            from PIL import Image

            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            frames: List[Any] = []
            timestamps: List[float] = []
            if total <= 0:
                # Fall back to sequential reading if frame count is unknown.
                ok, frame = cap.read()
                seq = []
                while ok:
                    seq.append(frame)
                    ok, frame = cap.read()
                total = len(seq)
                indices = self._even_indices(total, num_frames)
                for idx in indices:
                    rgb = cv2.cvtColor(seq[idx], cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb).convert("RGB"))
                    timestamps.append(idx / fps)
            else:
                indices = self._even_indices(total, num_frames)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb).convert("RGB"))
                    timestamps.append(idx / fps)
            cap.release()
            print(f"[CHEM][video_encoder] Read {len(frames)} frames via cv2.")
            return frames, timestamps
        except Exception as exc:
            print(f"[CHEM][video_encoder] cv2 unavailable/failed ({exc}); trying torchvision.")

        # --- Attempt 3: torchvision.io --------------------------------------
        try:
            import torchvision  # type: ignore
            from PIL import Image

            vframes, _aframes, info = torchvision.io.read_video(video_path, pts_unit="sec")
            total = int(vframes.shape[0])
            fps = float(info.get("video_fps", 25.0)) or 25.0
            if total == 0:
                return [], []
            indices = self._even_indices(total, num_frames)
            frames = [
                Image.fromarray(vframes[idx].numpy()).convert("RGB") for idx in indices
            ]
            timestamps = [idx / fps for idx in indices]
            print(f"[CHEM][video_encoder] Read {len(frames)} frames via torchvision.")
            return frames, timestamps
        except Exception as exc:
            print(f"[CHEM][video_encoder] torchvision unavailable/failed ({exc}).")

        print("[CHEM][video_encoder] No frame reader succeeded; returning no frames.")
        return [], []

    @staticmethod
    def _even_indices(total: int, num_frames: int) -> List[int]:
        """Return ``num_frames`` evenly spaced integer indices into ``total``.

        Inputs: total frame count and desired sample count.
        Outputs: a sorted list of in-range frame indices (length <= num_frames).
        """
        if total <= 0:
            return []
        if total <= num_frames:
            return list(range(total))
        step = total / float(num_frames)
        return [min(total - 1, int(round(i * step))) for i in range(num_frames)]

    # ------------------------------------------------------------------ action
    def _classify_action(self, frames: List[Any]) -> Tuple[str, float]:
        """Predict the dominant VideoMAE action label across the frames.

        VideoMAE expects a fixed number of frames (16); the sampled frames are
        padded/truncated to the processor's expectation.

        Inputs: a list of RGB PIL.Image frames.
        Outputs: ``(label, confidence)`` for the top-1 action.
        """
        torch = self._torch
        import numpy as np

        if not frames:
            return "unknown", 0.0

        # VideoMAE base was pretrained with 16 frames; normalise the count.
        needed = 16
        if len(frames) >= needed:
            sel = frames[:needed]
        else:
            sel = frames + [frames[-1]] * (needed - len(frames))

        arrays = [np.asarray(f) for f in sel]
        inputs = self._processor(arrays, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
        label = self._id2label.get(int(idx.item()), "unknown")
        return str(label), float(conf.item())

    def _map_action(self, label: str) -> Optional[str]:
        """Map a VideoMAE action label to a harm category, if any.

        Uses case-insensitive substring matching against the configured
        ``harmful_actions`` keys.

        Inputs: the predicted action label string.
        Outputs: a policy category id, or ``None`` if the action is benign.
        """
        low = (label or "").lower()
        for substring, category in self._harmful_actions.items():
            if substring.lower() in low:
                return category
        return None

    # ------------------------------------------------------------------ encode
    def encode(self, video_path: str, num_frames: int = 16) -> Dict[str, Any]:
        """Encode a video into per-frame + temporal-action CHEM elements.

        Inputs
        ------
        video_path : str
            Path to a readable video file.
        num_frames : int, default 16
            Number of frames to sample at regular intervals.

        Outputs
        -------
        dict
            ``{"elements": [...], "harm_score": float, "modality": "video"}``.
            Every element carries a ``temporal_position`` (seconds).
        """
        print(f"[CHEM][video_encoder] encode() path={video_path} num_frames={num_frames}")

        frames, timestamps = self._read_frames(video_path, num_frames)
        if not frames:
            print("[CHEM][video_encoder] No frames decoded; empty result.")
            return {"elements": [], "harm_score": 0.0, "modality": self.modality}

        elements: List[Dict[str, Any]] = []
        idx = 0
        harm_score = 0.0

        # --- Per-frame CLIP visual elements ---------------------------------
        for frame, ts in zip(frames, timestamps):
            try:
                frame_result = self._image_encoder.encode(frame)
            except Exception as exc:
                print(f"[CHEM][video_encoder] Frame CLIP failed at t={ts:.2f}s: {exc}")
                continue
            harm_score = max(harm_score, float(frame_result.get("harm_score", 0.0)))
            for iel in frame_result.get("elements", []):
                el = dict(iel)
                el["id"] = f"video_{idx}"
                el["type"] = "video"
                el["modality"] = self.modality
                el["temporal_position"] = float(ts)
                elements.append(el)
                idx += 1

        # --- Temporal VideoMAE action element -------------------------------
        try:
            action_label, action_conf = self._classify_action(frames)
        except Exception as exc:
            print(f"[CHEM][video_encoder] VideoMAE action classification failed: {exc}")
            action_label, action_conf = "unknown", 0.0

        category = self._map_action(action_label)
        is_harmful_action = category is not None
        # Position the action node at the temporal midpoint of the clip.
        mid_ts = float(timestamps[len(timestamps) // 2]) if timestamps else 0.0

        if is_harmful_action:
            harm_score = max(harm_score, action_conf)
        clause = get_policy_clause(category, self._config_path) if category else None

        # A VideoMAE feature vector isn't a natural 768-d embedding, so encode
        # the action node with a small numeric feature (confidence) lifted to
        # 768-d; the semantic category is carried in metadata.
        action_embedding = to_fixed_dim([action_conf])

        action_element: Dict[str, Any] = {
            "id": f"action_{idx}",
            "label": category if category else action_label,
            "type": "action",
            "embedding": action_embedding,
            "confidence": float(action_conf),
            "harmful_signal": bool(is_harmful_action),
            "full_label": (
                f"{clause.get('title', category)}: action '{action_label}'"
                if clause
                else f"action '{action_label}'"
            ),
            "modality": self.modality,
            "temporal_position": mid_ts,
        }
        if category:
            action_element["policy_category"] = category
        elements.append(action_element)
        idx += 1

        print(
            f"[CHEM][video_encoder] -> action='{action_label}' conf={action_conf:.4f} "
            f"category={category} harm_score={harm_score:.4f} elements={len(elements)}"
        )
        return {
            "elements": elements,
            "harm_score": float(harm_score),
            "modality": self.modality,
        }
