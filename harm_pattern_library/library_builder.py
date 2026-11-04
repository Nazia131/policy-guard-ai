"""Build the CHEM harm-pattern library from labelled training content.

This module is the orchestration layer of the harm-pattern library.  Given
labelled multimodal samples it:

1. Decomposes each sample into its element graph via
   :func:`graph.graph_builder.build_graph`.
2. Splits the resulting NetworkX graphs into harmful / safe pools by label.
3. Mines frequent, harm-discriminative subgraphs with :class:`SubgraphMiner`.
4. Tags each mined pattern with a policy category + clause via
   :class:`PatternTagger`.
5. Serialises the tagged patterns to ``harm_pattern_library/library.json``,
   where every entry carries the pattern subgraph, its policy clause, its
   support score and an example content id.

Downstream, :class:`model.harm_scorer.HarmScorer` loads this library and
matches new content graphs against these patterns.

The heavy sibling dependencies (``graph.graph_builder`` and the encoders) are
imported lazily inside the functions that need them, so importing this module
is cheap and does not fail if those siblings are still under construction.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from harm_pattern_library.pattern_tagger import PatternTagger
from harm_pattern_library.subgraph_miner import SubgraphMiner

# Default location of the emitted library, next to this file.
_DEFAULT_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "library.json")

# A "sample" is either ``(elements, label)`` or ``(elements, label, content_id)``.
Sample = Sequence[Any]


def build_library(
    samples: Optional[Iterable[Sample]] = None,
    loader: Optional[Callable[[], Iterable[Sample]]] = None,
    min_support: float = 0.3,
    max_nodes: int = 4,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Mine, tag and persist the harm-pattern library.

    Parameters
    ----------
    samples : iterable of sequence, optional
        An iterable of labelled samples.  Each sample is either
        ``(elements, label)`` or ``(elements, label, content_id)`` where
        ``elements`` is the list of canonical CHEM element dicts, ``label`` is
        ``1``/truthy for harmful and ``0``/falsey for safe, and the optional
        ``content_id`` is a human-readable identifier for the source content.
    loader : callable, optional
        A zero-argument callable returning such an iterable.  Used only when
        ``samples`` is not provided.
    min_support : float, optional
        Minimum harmful-support fraction for a mined pattern (default ``0.3``).
    max_nodes : int, optional
        Maximum node count of mined subgraphs (default ``4``).
    output_path : str, optional
        Where to write the library JSON (default
        ``harm_pattern_library/library.json``).
    config_path : str, optional
        Optional override for the policy configuration path.
    verbose : bool, optional
        When ``True`` (default) print ``[CHEM][library_builder]`` progress logs.

    Returns
    -------
    dict
        The in-memory library document that was written to disk, with keys
        ``"meta"`` and ``"patterns"``.  ``"patterns"`` is an empty list when no
        discriminative patterns are found or when no harmful graphs exist.
    """
    output_path = output_path or _DEFAULT_LIBRARY_PATH

    # ---- Resolve the sample source ------------------------------------- #
    if samples is None and loader is not None:
        if verbose:
            print("[CHEM][library_builder] Loading samples via provided loader.")
        samples = loader()
    if samples is None:
        raise ValueError(
            "build_library requires either 'samples' or 'loader' to be provided."
        )

    # ---- Decompose each sample into a graph and split by label --------- #
    build_graph = _import_build_graph()

    harmful_graphs: List[nx.Graph] = []
    safe_graphs: List[nx.Graph] = []
    n_seen = 0
    n_failed = 0

    for idx, sample in enumerate(samples):
        elements, label, content_id = _unpack_sample(sample, idx)
        n_seen += 1
        try:
            graph_dict = build_graph(elements, label)
            g = _extract_networkx(graph_dict)
        except Exception as exc:  # pragma: no cover - defensive per-sample guard
            n_failed += 1
            if verbose:
                print(
                    f"[CHEM][library_builder]   sample {idx} ({content_id}) "
                    f"failed to build a graph: {exc}"
                )
            continue

        # Record a content id on the graph for example attribution.
        g.graph.setdefault("content_id", content_id)

        if _is_harmful_label(label):
            harmful_graphs.append(g)
        else:
            safe_graphs.append(g)

        if verbose and (idx + 1) % 50 == 0:
            print(
                f"[CHEM][library_builder]   processed {idx + 1} samples "
                f"({len(harmful_graphs)} harmful / {len(safe_graphs)} safe)."
            )

    if verbose:
        print(
            f"[CHEM][library_builder] Built graphs for {n_seen - n_failed}/"
            f"{n_seen} samples: {len(harmful_graphs)} harmful, "
            f"{len(safe_graphs)} safe ({n_failed} failed)."
        )

    # ---- Mine + tag ---------------------------------------------------- #
    miner = SubgraphMiner(max_nodes=max_nodes, verbose=verbose)
    patterns = miner.mine(harmful_graphs, safe_graphs, min_support)

    tagger = PatternTagger(config_path=config_path, verbose=verbose)
    patterns = tagger.tag_all(patterns)

    # ---- Assemble + persist the library document ----------------------- #
    library = {
        "meta": {
            "name": "CHEM Harm Pattern Library",
            "min_support": float(min_support),
            "max_nodes": int(max_nodes),
            "num_patterns": len(patterns),
            "num_harmful_graphs": len(harmful_graphs),
            "num_safe_graphs": len(safe_graphs),
        },
        "patterns": [_library_entry(p) for p in patterns],
    }

    _write_json(library, output_path)
    if verbose:
        print(
            f"[CHEM][library_builder] Wrote {len(patterns)} patterns to "
            f"{output_path}."
        )
    return library


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _import_build_graph() -> Callable[..., Any]:
    """Import and return ``graph.graph_builder.build_graph`` lazily.

    Returns
    -------
    callable
        The ``build_graph(elements, label=None)`` convenience function.

    Raises
    ------
    ImportError
        If the graph-builder module cannot be imported, with a clear hint.
    """
    try:
        from graph.graph_builder import build_graph

        return build_graph
    except Exception as exc:  # pragma: no cover - depends on sibling module
        raise ImportError(
            "Could not import graph.graph_builder.build_graph; ensure the "
            "graph package is available and run from the project root."
        ) from exc


def _unpack_sample(sample: Sample, idx: int) -> Tuple[Any, Any, str]:
    """Normalise a sample into ``(elements, label, content_id)``.

    Parameters
    ----------
    sample : sequence or mapping
        A ``(elements, label)`` / ``(elements, label, content_id)`` tuple, or a
        mapping with ``elements``/``label``/``content_id`` keys.
    idx : int
        The running sample index, used to synthesise a content id when absent.

    Returns
    -------
    tuple
        ``(elements, label, content_id)``.

    Raises
    ------
    ValueError
        If the sample shape is not understood.
    """
    if isinstance(sample, dict):
        elements = sample.get("elements")
        label = sample.get("label")
        content_id = str(sample.get("content_id", f"sample_{idx}"))
        return elements, label, content_id

    if isinstance(sample, (tuple, list)):
        if len(sample) == 2:
            elements, label = sample
            return elements, label, f"sample_{idx}"
        if len(sample) >= 3:
            elements, label, content_id = sample[0], sample[1], sample[2]
            return elements, label, str(content_id)

    raise ValueError(
        f"Unrecognised sample at index {idx}: expected (elements, label[, "
        f"content_id]) or a mapping, got {type(sample).__name__}."
    )


def _extract_networkx(graph_dict: Any) -> nx.Graph:
    """Return the NetworkX graph from a ``build_graph`` result.

    Parameters
    ----------
    graph_dict : dict or networkx.Graph
        The result of :func:`graph.graph_builder.build_graph` (a dict with a
        ``"networkx"`` key), or a bare graph.

    Returns
    -------
    networkx.Graph
        The composition graph.

    Raises
    ------
    ValueError
        If no NetworkX graph can be located in the result.
    """
    if isinstance(graph_dict, (nx.Graph, nx.DiGraph)):
        return graph_dict
    if isinstance(graph_dict, dict):
        g = graph_dict.get("networkx") or graph_dict.get("graph")
        if isinstance(g, (nx.Graph, nx.DiGraph)):
            return g
    raise ValueError("build_graph result did not contain a NetworkX graph.")


def _is_harmful_label(label: Any) -> bool:
    """Return whether a label denotes harmful content.

    Parameters
    ----------
    label : any
        A sample label.  Numeric ``1`` / strings like ``"harmful"``/``"hateful"``
        and truthy values map to harmful; ``0``/``None``/``"safe"`` map to safe.

    Returns
    -------
    bool
        ``True`` for harmful, ``False`` for safe.
    """
    if label is None:
        return False
    if isinstance(label, str):
        return label.strip().lower() in {"1", "harmful", "hateful", "true", "yes"}
    try:
        return int(label) == 1
    except (TypeError, ValueError):
        return bool(label)


def _library_entry(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Project a tagged pattern into a stable library entry.

    Parameters
    ----------
    pattern : dict
        A tagged pattern from :meth:`PatternTagger.tag_all`.

    Returns
    -------
    dict
        A JSON-serialisable library entry containing the pattern subgraph, its
        policy category + clause, support scores, size and example content id.
    """
    return {
        "pattern_id": pattern.get("pattern_id"),
        "policy_category": pattern.get("policy_category"),
        "policy_clause": pattern.get("policy_clause"),
        "support": pattern.get("support"),
        "safe_support": pattern.get("safe_support"),
        "size": pattern.get("size"),
        "example_content_id": pattern.get("example_content_id"),
        "graph": pattern.get("graph"),
    }


def _write_json(document: Dict[str, Any], path: str) -> None:
    """Write ``document`` to ``path`` as pretty-printed JSON, creating dirs.

    Parameters
    ----------
    document : dict
        The JSON-serialisable library document.
    path : str
        Destination file path.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #
def _default_loader(
    dataset: str, split: str, limit: Optional[int], data_dir: Optional[str] = None
) -> Callable[[], Iterable[Sample]]:
    """Build a loader that yields encoded samples from a supported dataset.

    This wires the harm-pattern library to the project's data-preprocessing
    utilities lazily.  It intentionally raises an informative error rather than
    guessing when the requested dataset preprocessor is unavailable, so the
    failure mode is clear when running the CLI without prepared data.

    Parameters
    ----------
    dataset : str
        Dataset name (currently ``"hateful_memes"``).
    split : str
        Dataset split to load (e.g. ``"train"``).
    limit : int, optional
        Optional cap on the number of samples.
    data_dir : str, optional
        Dataset root directory (defaults to ``data/hateful_memes``).

    Returns
    -------
    callable
        A zero-argument loader yielding ``(elements, label, content_id)`` tuples.
    """

    def _loader() -> Iterable[Sample]:
        """Yield ``(elements, label, content_id)`` samples for the dataset."""
        if dataset != "hateful_memes":
            raise ValueError(
                f"No built-in loader for dataset '{dataset}'. Pass your own "
                f"'samples'/'loader' to build_library() instead."
            )
        try:
            from data.preprocess.hateful_memes import load_samples
        except Exception as exc:
            raise ImportError(
                "Could not import data.preprocess.hateful_memes.load_samples; "
                "prepare the dataset (see data/download_datasets.py) or supply "
                "your own samples to build_library()."
            ) from exc

        try:
            from encoders.text_encoder import TextEncoder
            from encoders.image_encoder import ImageEncoder
        except Exception as exc:
            raise ImportError(
                "Encoders are required to turn raw samples into elements; "
                "ensure encoders.text_encoder / encoders.image_encoder import."
            ) from exc

        text_encoder = TextEncoder()
        image_encoder = ImageEncoder()

        count = 0
        for raw in load_samples(split=split, data_dir=data_dir):
            if limit is not None and count >= limit:
                break
            elements: List[Dict[str, Any]] = []
            text = raw.get("text")
            if text:
                elements.extend(text_encoder.encode(text).get("elements", []))
            image = raw.get("img") or raw.get("image")
            if image:
                elements.extend(image_encoder.encode(image).get("elements", []))
            content_id = str(raw.get("id", f"{dataset}_{count}"))
            yield elements, raw.get("label", 0), content_id
            count += 1

    return _loader


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for building the harm-pattern library.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv``).

    Returns
    -------
    int
        Process exit code (``0`` on success, non-zero on handled failure).
    """
    parser = argparse.ArgumentParser(
        description="Build the CHEM harm-pattern library from labelled content.",
    )
    parser.add_argument(
        "--dataset",
        default="hateful_memes",
        help="Dataset to mine patterns from (default: hateful_memes).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples to process.",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.3,
        help="Minimum harmful-support fraction for a pattern (default: 0.3).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=4,
        help="Maximum number of nodes per mined subgraph (default: 4).",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default=_DEFAULT_LIBRARY_PATH,
        help="Output path for the library JSON.",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Dataset root directory (default: data/hateful_memes).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional override path for policy_config.json.",
    )
    args = parser.parse_args(argv)

    print("[CHEM][library_builder] Starting harm-pattern library build.")
    try:
        loader = _default_loader(args.dataset, args.split, args.limit, args.data_dir)
        build_library(
            loader=loader,
            min_support=args.min_support,
            max_nodes=args.max_nodes,
            output_path=args.output,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"[CHEM][library_builder] Build failed: {exc}")
        return 1
    print("[CHEM][library_builder] Build complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
