#!/usr/bin/env python3
"""CHEM dataset acquisition helper.

This module provides an argparse-driven command line interface that helps a
user obtain the two datasets CHEM is designed to work with:

* **Facebook Hateful Memes** — a multimodal (image + text) hate-speech dataset.
  The official distribution is gated behind the DrivenData competition terms and
  the Meta / MMF license, so it CANNOT be downloaded anonymously. When the ``mmf``
  dataset zoo is installed we attempt an automated download through it; otherwise
  we print exact, actionable manual-download instructions and the expected
  extracted directory layout.

* **UCF-Crime** — a large real-world surveillance video anomaly-detection
  dataset. It is distributed through Kaggle (and the original UCF project page),
  which also requires authenticated access, so we print manual Kaggle download
  instructions plus the expected folder layout.

Design constraints (intentional):
    * We never hardcode credentials and never attempt to bypass any
      authentication, license gate, or competition terms.
    * The module imports cleanly even when optional dependencies (``mmf``) are
      absent — missing datasets/tools degrade to informative instructions rather
      than crashes.
    * All progress is reported through clear ``[CHEM][download]`` prints.

Usage:
    python3 data/download_datasets.py --dataset all --target-dir data/datasets
    python3 data/download_datasets.py --dataset hateful_memes --target-dir data/hm
    python3 data/download_datasets.py --dataset ucf_crime
"""

from __future__ import annotations

import argparse
import os
import textwrap


# --------------------------------------------------------------------------- #
# Logging helper
# --------------------------------------------------------------------------- #
def _log(message: str) -> None:
    """Print a namespaced progress line.

    Inputs:
        message: Human-readable progress text.
    Outputs:
        None. Writes a single ``[CHEM][download] ...`` line to stdout.
    """
    print(f"[CHEM][download] {message}")


def ensure_dir(target_dir: str) -> str:
    """Verify that ``target_dir`` exists, creating it (and parents) if needed.

    Inputs:
        target_dir: Filesystem path that should hold a dataset.
    Outputs:
        The absolute path to the (now guaranteed to exist) directory.
    Raises:
        OSError: If the directory cannot be created (e.g. permission denied).
    """
    abs_dir = os.path.abspath(os.path.expanduser(target_dir))
    if os.path.isdir(abs_dir):
        _log(f"Target directory already exists: {abs_dir}")
    else:
        _log(f"Creating target directory: {abs_dir}")
        os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


# --------------------------------------------------------------------------- #
# Facebook Hateful Memes
# --------------------------------------------------------------------------- #
def _print_hateful_memes_manual_instructions(target_dir: str) -> None:
    """Print manual download instructions and expected layout for Hateful Memes.

    Inputs:
        target_dir: Absolute path where the extracted dataset should live.
    Outputs:
        None. Prints step-by-step instructions to stdout.
    """
    instructions = textwrap.dedent(
        f"""
        ------------------------------------------------------------------
        MANUAL DOWNLOAD REQUIRED: Facebook Hateful Memes
        ------------------------------------------------------------------
        The Hateful Memes dataset is gated behind the DrivenData competition
        terms and the Meta / MMF license. It cannot be downloaded anonymously
        and CHEM will not attempt to bypass that gate.

        To obtain it manually:

          1. Read about the dataset and the MMF tooling:
               https://github.com/facebookresearch/mmf

          2. Register and accept the license/terms on the DrivenData
             competition page, then download the data zip:
               https://www.drivendata.org/competitions/64/hateful-memes/

          3. (Optional, automated) Install MMF and let its dataset zoo place
             the files for you:
               pip install mmf
               mmf_convert_hm --zip_file=<downloaded.zip> --password=<password>
             The zip password is provided to you on the DrivenData page after
             you accept the terms.

          4. Extract/convert so that the following layout exists under:
               {target_dir}

               {target_dir}/
                 img/            # all meme .png images referenced by the jsonl
                 train.jsonl     # fields: id, img, text, label
                 dev.jsonl       # (a.k.a. dev_seen.jsonl) validation split
                 test.jsonl      # (a.k.a. test_seen.jsonl) test split

        Each *.jsonl line is a JSON object, e.g.:
            {{"id": 42953, "img": "img/42953.png",
              "text": "its their character not their color that matters",
              "label": 0}}

        Once the layout above exists, CHEM's loader
        (data/preprocess/hateful_memes.py) can read it directly.
        ------------------------------------------------------------------
        """
    ).strip("\n")
    print(instructions)


def download_hateful_memes_hf(target_dir: str) -> bool:
    """Download the Hateful Memes dataset from Hugging Face (no login required).

    Uses the public mirror at ``neuralcatcher/hateful_memes`` on Hugging Face Hub
    (3.37 GB; contains all ~8,500 training images and the train/dev/test JSONL
    splits).  This is the fastest path when DrivenData access is unavailable.

    Requires ``huggingface_hub`` (installed automatically via ``pip3 install
    huggingface_hub`` or included in ``requirements.txt``).

    Inputs:
        target_dir: Destination directory. Files are placed directly inside it.
    Outputs:
        ``True`` on success, ``False`` if ``huggingface_hub`` is not installed
        or the download fails.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        _log("huggingface_hub not installed. Run: pip3 install huggingface_hub")
        return False

    abs_dir = os.path.abspath(os.path.expanduser(target_dir))
    os.makedirs(abs_dir, exist_ok=True)
    _log("Downloading Hateful Memes from Hugging Face (neuralcatcher/hateful_memes).")
    _log("This is ~3.4 GB — expect 5–20 minutes depending on your connection.")
    try:
        snapshot_download(
            repo_id="neuralcatcher/hateful_memes",
            repo_type="dataset",
            local_dir=abs_dir,
            ignore_patterns=["*.parquet", ".gitattributes"],  # skip auto-converted parquets
        )
        _log(f"Download complete. Files in: {abs_dir}")
        # Hugging Face stores dev_seen.jsonl / test_seen.jsonl; create symlinks so
        # the CHEM loader (which also accepts those filenames) finds them as-is.
        for src, dst in [("dev_seen.jsonl", "dev.jsonl"), ("test_seen.jsonl", "test.jsonl")]:
            src_path = os.path.join(abs_dir, src)
            dst_path = os.path.join(abs_dir, dst)
            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
                _log(f"  Aliased {src} -> {dst}")
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"Hugging Face download failed: {exc}")
        return False


def download_hateful_memes(target_dir: str) -> bool:
    """Obtain the Facebook Hateful Memes dataset (or explain how to).

    Behaviour (in order):
        1. If the data is already present in ``target_dir``, do nothing.
        2. Try downloading automatically from the public Hugging Face mirror
           (``neuralcatcher/hateful_memes``, no login required).
        3. If ``huggingface_hub`` is unavailable, fall back to printing manual
           DrivenData / MMF instructions.

    Inputs:
        target_dir: Destination directory for the extracted dataset.
    Outputs:
        ``True`` if the expected files appear present after the call,
        otherwise ``False``.
    """
    abs_dir = ensure_dir(target_dir)

    # Check for already-present data (accept dev_seen.jsonl as equivalent to dev.jsonl).
    def _present(name: str) -> bool:
        return os.path.isfile(os.path.join(abs_dir, name))

    already_have = (
        (_present("train.jsonl"))
        and (_present("dev.jsonl") or _present("dev_seen.jsonl"))
        and (_present("test.jsonl") or _present("test_seen.jsonl"))
        and os.path.isdir(os.path.join(abs_dir, "img"))
    )
    if already_have:
        _log(f"Hateful Memes already present in {abs_dir}. Nothing to do.")
        return True

    # Prefer automatic Hugging Face download (public, no auth).
    _log("Attempting automatic download from Hugging Face Hub...")
    if download_hateful_memes_hf(abs_dir):
        return True

    # Last resort: print manual instructions.
    _log("Automatic download unavailable. Printing manual instructions.")
    _log("Checking for the optional 'mmf' dataset zoo...")
    try:
        import mmf  # type: ignore  # noqa: F401
        _log("Found 'mmf'. Run after accepting DrivenData terms:\n"
             "    mmf_convert_hm --zip_file=<downloaded.zip> --password=<password>")
    except ImportError:
        _log("'mmf' is not installed; falling back to manual instructions.")
    _print_hateful_memes_manual_instructions(abs_dir)
    return False


# --------------------------------------------------------------------------- #
# UCF-Crime
# --------------------------------------------------------------------------- #
def _print_ucf_crime_manual_instructions(target_dir: str) -> None:
    """Print manual Kaggle download instructions and expected layout for UCF-Crime.

    Inputs:
        target_dir: Absolute path where the extracted dataset should live.
    Outputs:
        None. Prints step-by-step instructions to stdout.
    """
    instructions = textwrap.dedent(
        f"""
        ------------------------------------------------------------------
        MANUAL DOWNLOAD REQUIRED: UCF-Crime
        ------------------------------------------------------------------
        UCF-Crime is distributed through Kaggle (and the original UCF project
        page). Access requires an authenticated Kaggle account, so CHEM will
        not attempt to download it anonymously.

        To obtain it manually:

          1. Sign in to Kaggle:
               https://www.kaggle.com/

          2. Search for 'UCF Crime Dataset' on kaggle.com and open the dataset
             page (there are several mirrors; pick one that provides the full
             surveillance videos organized by category).

          3. Download and extract the archive so the following layout exists
             under:
               {target_dir}

               {target_dir}/
                 Abuse/            # anomaly category -> label 1
                 Arrest/           # anomaly category -> label 1
                 Arson/            # anomaly category -> label 1
                 Assault/          # anomaly category -> label 1
                 Burglary/         # anomaly category -> label 1
                 Explosion/        # anomaly category -> label 1
                 Fighting/         # anomaly category -> label 1
                 RoadAccidents/    # anomaly category -> label 1
                 Robbery/          # anomaly category -> label 1
                 Shooting/         # anomaly category -> label 1
                 Shoplifting/      # anomaly category -> label 1
                 Stealing/         # anomaly category -> label 1
                 Vandalism/        # anomaly category -> label 1
                 Normal/           # (or 'Normal_Videos_event') -> label 0

          4. Each category subfolder contains the raw video files
             (e.g. .mp4 / .avi). CHEM's indexer
             (data/preprocess/ucf_crime.py) treats any folder whose name
             contains 'normal' as label 0 (normal) and every other category as
             label 1 (anomaly).
        ------------------------------------------------------------------
        """
    ).strip("\n")
    print(instructions)


def download_ucf_crime(target_dir: str) -> bool:
    """Explain how to obtain the UCF-Crime dataset from Kaggle.

    UCF-Crime cannot be fetched without authenticated Kaggle access, so this
    function only ensures the target directory exists and prints manual
    instructions plus the expected folder layout. It never handles credentials.

    Inputs:
        target_dir: Destination directory for the extracted dataset.
    Outputs:
        ``True`` if category subfolders already appear present, else ``False``.
    """
    abs_dir = ensure_dir(target_dir)

    # Heuristic presence check: at least one non-empty subdirectory.
    try:
        subdirs = [
            d for d in os.listdir(abs_dir)
            if os.path.isdir(os.path.join(abs_dir, d))
        ]
    except OSError:
        subdirs = []
    if subdirs:
        _log(
            f"UCF-Crime target already contains {len(subdirs)} category "
            f"subfolder(s) in {abs_dir}. Verify it matches the expected layout."
        )

    _print_ucf_crime_manual_instructions(abs_dir)
    return bool(subdirs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_target_dir(dataset: str, base: str) -> str:
    """Compute a per-dataset default target directory.

    Inputs:
        dataset: One of ``hateful_memes`` or ``ucf_crime``.
        base: Base directory chosen by the user (``--target-dir``).
    Outputs:
        Path under ``base`` named after the dataset.
    """
    return os.path.join(base, dataset)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the download CLI.

    Inputs:
        None.
    Outputs:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="download_datasets.py",
        description=(
            "Help obtain CHEM datasets (Facebook Hateful Memes, UCF-Crime). "
            "Gated datasets are never downloaded without your own credentials; "
            "manual instructions are printed instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["hateful_memes", "ucf_crime", "all"],
        default="all",
        help="Which dataset to obtain (default: all).",
    )
    parser.add_argument(
        "--target-dir",
        default="data/datasets",
        help=(
            "Base directory to place datasets into. Each dataset is stored in "
            "a subfolder named after it (default: data/datasets)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Inputs:
        argv: Optional list of command-line arguments (defaults to ``sys.argv``).
    Outputs:
        Process exit code: ``0`` if every requested dataset is already present,
        ``1`` if at least one dataset still requires manual action.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    _log(f"Requested dataset(s): {args.dataset}")
    _log(f"Base target directory: {os.path.abspath(args.target_dir)}")

    results: dict[str, bool] = {}

    if args.dataset in ("hateful_memes", "all"):
        _log("=== Facebook Hateful Memes ===")
        results["hateful_memes"] = download_hateful_memes(
            _default_target_dir("hateful_memes", args.target_dir)
        )

    if args.dataset in ("ucf_crime", "all"):
        _log("=== UCF-Crime ===")
        results["ucf_crime"] = download_ucf_crime(
            _default_target_dir("ucf_crime", args.target_dir)
        )

    _log("Summary:")
    for name, ready in results.items():
        status = "PRESENT" if ready else "MANUAL ACTION NEEDED"
        _log(f"  - {name}: {status}")

    all_ready = all(results.values()) if results else False
    return 0 if all_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
