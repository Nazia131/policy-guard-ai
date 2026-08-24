#!/usr/bin/env python3
"""Indexing utilities for the UCF-Crime video anomaly dataset.

UCF-Crime organises raw surveillance videos into one subfolder per category.
Every category except the "Normal" videos denotes an anomalous (harmful) event:

    Abuse/ Arrest/ Arson/ Assault/ Burglary/ Explosion/ Fighting/
    RoadAccidents/ Robbery/ Shooting/ Shoplifting/ Stealing/ Vandalism/
      -> anomaly, label 1
    Normal/ (a.k.a. Normal_Videos_event/, Testing_Normal_Videos, ...)
      -> normal,  label 0

This module scans that structure and produces a flat index of samples. It uses
only the standard library and imports cleanly even when the dataset is absent;
:func:`index_ucf_crime` raises a single informative error if the directory is
missing.

Exposes:
    * :func:`index_ucf_crime` — return a list of sample dicts.
    * :class:`UCFCrimeDataset` — an index-able wrapper over that list.
"""

from __future__ import annotations

import os
from typing import Any, Iterator


# Video file extensions recognised when scanning category folders.
_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")


def _log(message: str) -> None:
    """Print a namespaced progress line.

    Inputs:
        message: Human-readable progress text.
    Outputs:
        None. Writes ``[CHEM][ucf_crime] ...`` to stdout.
    """
    print(f"[CHEM][ucf_crime] {message}")


def _is_normal_category(category: str) -> bool:
    """Decide whether a category folder name denotes normal (non-anomalous) video.

    The UCF-Crime distribution names its benign videos with variations such as
    ``Normal``, ``Normal_Videos_event`` or ``Testing_Normal_Videos``; any folder
    whose name contains the substring "normal" (case-insensitive) is treated as
    label 0.

    Inputs:
        category: The category subfolder name.
    Outputs:
        ``True`` if the category is a normal/benign category, else ``False``.
    """
    return "normal" in category.lower()


def index_ucf_crime(data_dir: str) -> list[dict[str, Any]]:
    """Scan the UCF-Crime directory and build a flat sample index.

    Walks each immediate subdirectory of ``data_dir`` (each is a category),
    collects the video files inside it, and assigns a binary label based on
    whether the category name denotes normal or anomalous footage.

    Inputs:
        data_dir: UCF-Crime root directory containing category subfolders.
    Outputs:
        A list of dicts, one per video, of the form::

            {
              "id":       str,  # "<category>/<filename>", unique per video
              "video":    str,  # ABSOLUTE path to the video file
              "category": str,  # the category subfolder name (e.g. "Fighting")
              "label":    int,  # 0 for normal, 1 for anomaly
            }

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist, with guidance on how
            to obtain the dataset.
    """
    abs_root = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(abs_root):
        raise FileNotFoundError(
            f"UCF-Crime data directory not found: {abs_root}\n"
            "Obtain the dataset first (see data/download_datasets.py: "
            "`python3 data/download_datasets.py --dataset ucf_crime`)."
        )

    _log(f"Indexing UCF-Crime videos under {abs_root}")

    samples: list[dict[str, Any]] = []
    categories = sorted(
        d for d in os.listdir(abs_root) if os.path.isdir(os.path.join(abs_root, d))
    )
    if not categories:
        _log("Warning: no category subfolders found. Is the dataset extracted?")

    for category in categories:
        category_path = os.path.join(abs_root, category)
        label = 0 if _is_normal_category(category) else 1

        video_count = 0
        # Walk recursively so nested layouts (e.g. Anomaly/Fighting/...) work too.
        for dirpath, _dirnames, filenames in os.walk(category_path):
            for filename in sorted(filenames):
                if not filename.lower().endswith(_VIDEO_EXTENSIONS):
                    continue
                abs_video = os.path.join(dirpath, filename)
                rel_id = os.path.relpath(abs_video, abs_root)
                samples.append(
                    {
                        "id": rel_id,
                        "video": abs_video,
                        "category": category,
                        "label": label,
                    }
                )
                video_count += 1

        kind = "normal" if label == 0 else "anomaly"
        _log(f"  {category}: {video_count} video(s) [{kind}, label={label}]")

    n_anomaly = sum(1 for s in samples if s["label"] == 1)
    n_normal = len(samples) - n_anomaly
    _log(
        f"Indexed {len(samples)} videos across {len(categories)} categories "
        f"({n_anomaly} anomaly, {n_normal} normal)."
    )
    return samples


class UCFCrimeDataset:
    """Index-able wrapper around the UCF-Crime video index.

    Builds the flat sample index once at construction and exposes a minimal
    sequence protocol (``__len__`` / ``__getitem__`` / iteration) so it can be
    used like a PyTorch ``Dataset``.

    Attributes:
        data_dir: UCF-Crime root directory.
        samples: List of sample dicts (see :func:`index_ucf_crime`).
    """

    def __init__(self, data_dir: str) -> None:
        """Build the index for a UCF-Crime directory.

        Inputs:
            data_dir: UCF-Crime root directory containing category subfolders.
        Raises:
            FileNotFoundError: If the data directory is missing.
        """
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.samples: list[dict[str, Any]] = index_ucf_crime(self.data_dir)
        _log(f"UCFCrimeDataset ready: {len(self.samples)} videos.")

    def __len__(self) -> int:
        """Return the number of indexed videos.

        Outputs:
            Integer video count.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the sample at ``index``.

        Inputs:
            index: Zero-based position into the index.
        Outputs:
            The sample dict at ``index``.
        Raises:
            IndexError: If ``index`` is out of range.
        """
        return self.samples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over indexed videos in order.

        Outputs:
            An iterator over the sample dicts.
        """
        return iter(self.samples)


if __name__ == "__main__":
    # Tiny smoke-test CLI: index a directory and print a few entries.
    import argparse

    _parser = argparse.ArgumentParser(description="Index a UCF-Crime directory.")
    _parser.add_argument("data_dir", help="UCF-Crime root directory.")
    _parser.add_argument("--n", type=int, default=5, help="How many entries to show.")
    _args = _parser.parse_args()

    _ds = UCFCrimeDataset(_args.data_dir)
    for _i in range(min(_args.n, len(_ds))):
        print(_ds[_i])
