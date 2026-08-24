"""Content Composition Graph construction for CHEM.

The heart of CHEM (Compositional Harm Emergence Modeling) is the idea that harm
often *emerges* from the composition of individually-benign elements: a benign
caption over a benign image, a calm transcript over an agitated tone of voice, a
person plus a weapon plus a pointing relation.  To reason about that composition
we turn a decomposed piece of multimodal content into a directed graph -- the
*Content Composition Graph* -- and hand it to a Graph Attention Network.

This module takes the canonical *element* dicts emitted by the CHEM encoders
(see the shared interface contract) and builds that graph, producing:

* a ``torch_geometric.data.Data`` object for the model,
* a NetworkX ``DiGraph`` for visualisation / pattern mining,
* JSON-serialisable ``nodes`` / ``edges`` lists for the API and frontend.

Five edge types are built, exactly per the CHEM spec (ids come from
``policy/policy_config.json`` via :mod:`graph.edge_types`):

* ``co_occurrence`` (0)      -- same-modality elements adjacent in extraction
  order, within ``co_occurrence_window`` positions.
* ``semantic_reference`` (1) -- elements of *different* modalities.
* ``temporal_sequence`` (2)  -- consecutive video / audio / action elements
  ordered by ``temporal_position``.
* ``target_pointing`` (3)    -- a harmful text element pointing at a visual
  subject (``target_id`` if present, else the nearest image/video element).
* ``harmful_pattern`` (4)    -- overrides any existing edge whose *both*
  endpoints carry ``harmful_signal=True``; tracked as its own edge type.

No harm semantics or numeric thresholds are hardcoded here; the co-occurrence
window is read from the policy config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Heavy numeric dependencies.  These are genuine runtime dependencies of CHEM;
# it is fine if they are not installed in an environment that only AST-parses
# this file, because the imports are only exercised when the graph is built.
import torch
import networkx as nx
from torch_geometric.data import Data

from utils.config import get_coefficient
from utils.embeddings import stack_fixed_dim, target_dim
from graph.edge_types import EDGE_TYPES, edge_type_name


# Element ``type`` values that participate in the temporal sequence.  These are
# the inherently time-ordered modalities.  (Not harm semantics -- structural.)
_TEMPORAL_TYPES = ("video", "audio", "action")

# Element ``type``/``modality`` values that can serve as a visual subject for a
# text->visual ``target_pointing`` edge.
_VISUAL_TYPES = ("image", "video")


def _log(message: str) -> None:
    """Print a namespaced progress log line for the graph builder.

    Parameters
    ----------
    message : str
        The message to print (prefixed with ``[CHEM][graph_builder]``).

    Returns
    -------
    None
    """
    print(f"[CHEM][graph_builder] {message}")


class ContentCompositionGraphBuilder:
    """Build a Content Composition Graph from decomposed content elements.

    The builder is stateless apart from the cached co-occurrence window read
    from the policy config in :meth:`__init__`, so a single instance can be
    reused across many pieces of content.

    Parameters
    ----------
    config_path : str, optional
        Optional override for the policy config path.  Defaults to the bundled
        ``policy/policy_config.json``.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the builder and cache config-driven parameters.

        Parameters
        ----------
        config_path : str, optional
            Optional override for the policy config path.

        Returns
        -------
        None
        """
        self._config_path = config_path
        # Structural window governing co-occurrence adjacency.  Read once.
        self._co_occurrence_window = int(
            get_coefficient("co_occurrence_window", config_path)
        )
        self._feature_dim = target_dim(config_path)
        _log(
            "Initialised ContentCompositionGraphBuilder "
            f"(co_occurrence_window={self._co_occurrence_window}, "
            f"feature_dim={self._feature_dim})"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(
        self,
        elements: List[Dict[str, Any]],
        label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the Content Composition Graph for a list of elements.

        Parameters
        ----------
        elements : list of dict
            Canonical CHEM element dicts (see the interface contract).  Each
            must carry at least ``id``, ``type``, ``modality`` and a 768-dim
            ``embedding``.  An empty list yields a valid empty graph.
        label : int, optional
            Ground-truth graph label (``0`` safe / ``1`` harmful).  When given
            it is attached to the PyG ``Data`` as ``y = LongTensor([label])``.

        Returns
        -------
        dict
            ``{"data": torch_geometric.data.Data, "nodes": list, "edges": list,
               "networkx": networkx.DiGraph}`` where ``nodes``/``edges`` are
            JSON-serialisable and ``data`` also carries a ``node_meta`` list.
        """
        elements = elements or []
        n = len(elements)
        _log(f"Building graph from {n} element(s)"
             + (f" with label={label}" if label is not None else ""))

        # ---- Node feature matrix ------------------------------------- #
        # Project every embedding to the fixed feature dim (768).
        embeddings = [el.get("embedding", []) for el in elements]
        feature_rows = stack_fixed_dim(embeddings, self._feature_dim)
        x = torch.tensor(feature_rows, dtype=torch.float).reshape(n, self._feature_dim)

        # ---- Node metadata (JSON-serialisable) ----------------------- #
        node_meta = [self._node_metadata(idx, el) for idx, el in enumerate(elements)]

        # ---- Edge construction --------------------------------------- #
        # ``edge_map`` collapses to a single relation per directed pair so the
        # harmful_pattern override has clean semantics.  Keyed (src, dst).
        edge_map: Dict[Tuple[int, int], int] = {}
        self._add_co_occurrence_edges(elements, edge_map)
        self._add_semantic_reference_edges(elements, edge_map)
        self._add_temporal_sequence_edges(elements, edge_map)
        self._add_target_pointing_edges(elements, edge_map)
        self._apply_harmful_pattern_override(elements, edge_map)

        # ---- Assemble PyG tensors ------------------------------------ #
        if edge_map:
            src_dst = list(edge_map.keys())
            edge_index = torch.tensor(src_dst, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(list(edge_map.values()), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        # Attach lightweight metadata so downstream code (explainers, API) can
        # recover human-readable node info without the feature tensor.
        data.node_meta = node_meta
        if label is not None:
            data.y = torch.tensor([int(label)], dtype=torch.long)

        # ---- JSON-serialisable views + NetworkX graph ---------------- #
        nodes_json = node_meta  # already tensor-free
        edges_json = [
            {
                "source": src,
                "target": dst,
                "edge_type": etype,
                "edge_type_name": edge_type_name(etype),
            }
            for (src, dst), etype in edge_map.items()
        ]
        graph_nx = self._to_networkx(node_meta, edge_map)

        _log(
            f"Built graph: {n} node(s), {len(edges_json)} edge(s) "
            f"[{self._edge_type_summary(edge_map)}]"
        )
        return {
            "data": data,
            "nodes": nodes_json,
            "edges": edges_json,
            "networkx": graph_nx,
        }

    # ------------------------------------------------------------------ #
    # Node helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _node_metadata(index: int, element: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a JSON-serialisable metadata dict for one element.

        The full 768-dim embedding is intentionally omitted (it lives in the
        feature tensor) so the returned dict stays lightweight for the API and
        frontend.

        Parameters
        ----------
        index : int
            The node's positional index in the graph (its row in ``x``).
        element : dict
            The canonical element dict.

        Returns
        -------
        dict
            Tensor-free metadata for the node.
        """
        meta = {
            "index": index,
            "id": element.get("id", f"node_{index}"),
            "label": element.get("label"),
            "full_label": element.get("full_label"),
            "type": element.get("type"),
            "modality": element.get("modality"),
            "confidence": element.get("confidence"),
            "harmful_signal": bool(element.get("harmful_signal", False)),
        }
        # Carry optional, JSON-friendly fields through when present.
        for optional_key in (
            "policy_category",
            "temporal_position",
            "target_id",
            "text",
            "language",
        ):
            if optional_key in element:
                meta[optional_key] = element[optional_key]
        return meta

    # ------------------------------------------------------------------ #
    # Edge builders (each mutates ``edge_map`` in place)
    # ------------------------------------------------------------------ #
    def _add_co_occurrence_edges(
        self,
        elements: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Add ``co_occurrence`` (0) edges between adjacent same-modality nodes.

        Two elements co-occur when they share a modality and lie within
        ``co_occurrence_window`` positions of each other in extraction order.

        Parameters
        ----------
        elements : list of dict
            The graph elements in extraction order.
        edge_map : dict
            ``(src, dst) -> edge_type_id`` accumulator, mutated in place.

        Returns
        -------
        None
        """
        etype = EDGE_TYPES["co_occurrence"]
        window = self._co_occurrence_window
        n = len(elements)
        for i in range(n):
            mod_i = elements[i].get("modality")
            for j in range(i + 1, min(i + window + 1, n)):
                if elements[j].get("modality") == mod_i:
                    edge_map[(i, j)] = etype

    @staticmethod
    def _add_semantic_reference_edges(
        elements: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Add ``semantic_reference`` (1) edges across different modalities.

        Every ordered pair of elements whose modalities differ is linked, in the
        forward extraction direction, capturing cross-modal reference (e.g. a
        caption referring to an image).

        Parameters
        ----------
        elements : list of dict
            The graph elements in extraction order.
        edge_map : dict
            ``(src, dst) -> edge_type_id`` accumulator, mutated in place.

        Returns
        -------
        None
        """
        etype = EDGE_TYPES["semantic_reference"]
        n = len(elements)
        for i in range(n):
            mod_i = elements[i].get("modality")
            for j in range(i + 1, n):
                if elements[j].get("modality") != mod_i:
                    # Do not clobber a previously-set relation for this pair.
                    edge_map.setdefault((i, j), etype)

    @staticmethod
    def _add_temporal_sequence_edges(
        elements: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Add ``temporal_sequence`` (2) edges between time-ordered elements.

        Video / audio / action elements carrying a ``temporal_position`` are
        sorted by timestamp and consecutive elements are linked, forming a
        temporal chain.  Overwrites any earlier relation on the same directed
        pair because temporal ordering is the more specific structural relation.

        Parameters
        ----------
        elements : list of dict
            The graph elements in extraction order.
        edge_map : dict
            ``(src, dst) -> edge_type_id`` accumulator, mutated in place.

        Returns
        -------
        None
        """
        etype = EDGE_TYPES["temporal_sequence"]
        timed = [
            (idx, el.get("temporal_position"))
            for idx, el in enumerate(elements)
            if el.get("type") in _TEMPORAL_TYPES and el.get("temporal_position") is not None
        ]
        # Sort by timestamp; stable on the original index for ties.
        timed.sort(key=lambda pair: (float(pair[1]), pair[0]))
        for (src, _), (dst, _) in zip(timed, timed[1:]):
            edge_map[(src, dst)] = etype

    @staticmethod
    def _add_target_pointing_edges(
        elements: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Add ``target_pointing`` (3) edges from harmful text to a visual subject.

        For every text element flagged ``harmful_signal=True`` we draw an edge to
        the visual subject it targets: the element named by ``target_id`` if it
        resolves to an image/video node, otherwise the nearest image/video
        element by extraction-order distance.

        Parameters
        ----------
        elements : list of dict
            The graph elements in extraction order.
        edge_map : dict
            ``(src, dst) -> edge_type_id`` accumulator, mutated in place.

        Returns
        -------
        None
        """
        etype = EDGE_TYPES["target_pointing"]
        # Index elements by id for fast target_id resolution.
        id_to_index = {el.get("id"): idx for idx, el in enumerate(elements)}
        visual_indices = [
            idx for idx, el in enumerate(elements) if el.get("type") in _VISUAL_TYPES
        ]
        if not visual_indices:
            return

        for i, el in enumerate(elements):
            if el.get("type") != "text" or not el.get("harmful_signal", False):
                continue

            target_index: Optional[int] = None
            target_id = el.get("target_id")
            if target_id is not None and target_id in id_to_index:
                candidate = id_to_index[target_id]
                if elements[candidate].get("type") in _VISUAL_TYPES:
                    target_index = candidate

            if target_index is None:
                # Nearest visual element by absolute positional distance.
                target_index = min(visual_indices, key=lambda vi: (abs(vi - i), vi))

            if target_index != i:
                edge_map[(i, target_index)] = etype

    @staticmethod
    def _apply_harmful_pattern_override(
        elements: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Relabel edges whose both endpoints are harmful as ``harmful_pattern`` (4).

        This is the compositional core of CHEM: wherever two elements that each
        carry a harmful signal are already connected by any relation, that edge
        is promoted to the ``harmful_pattern`` type, marking an emergent harmful
        composition.  The override is applied last so it wins over every other
        relation type.

        Parameters
        ----------
        elements : list of dict
            The graph elements in extraction order.
        edge_map : dict
            ``(src, dst) -> edge_type_id`` accumulator, mutated in place.

        Returns
        -------
        None
        """
        etype = EDGE_TYPES["harmful_pattern"]
        harmful = [bool(el.get("harmful_signal", False)) for el in elements]
        for (src, dst) in list(edge_map.keys()):
            if harmful[src] and harmful[dst]:
                edge_map[(src, dst)] = etype

    # ------------------------------------------------------------------ #
    # NetworkX + reporting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_networkx(
        node_meta: List[Dict[str, Any]],
        edge_map: Dict[Tuple[int, int], int],
    ) -> "nx.DiGraph":
        """Build a NetworkX ``DiGraph`` mirroring the graph structure.

        Parameters
        ----------
        node_meta : list of dict
            Per-node metadata (used as node attributes).
        edge_map : dict
            ``(src, dst) -> edge_type_id`` mapping.

        Returns
        -------
        networkx.DiGraph
            Directed graph with node attributes copied from ``node_meta`` and an
            ``edge_type`` / ``edge_type_name`` attribute on every edge.
        """
        graph = nx.DiGraph()
        for meta in node_meta:
            graph.add_node(meta["index"], **meta)
        for (src, dst), etype in edge_map.items():
            graph.add_edge(
                src,
                dst,
                edge_type=etype,
                edge_type_name=edge_type_name(etype),
            )
        return graph

    @staticmethod
    def _edge_type_summary(edge_map: Dict[Tuple[int, int], int]) -> str:
        """Summarise edge-type counts for a progress log.

        Parameters
        ----------
        edge_map : dict
            ``(src, dst) -> edge_type_id`` mapping.

        Returns
        -------
        str
            A compact ``"name=count"`` summary, e.g. ``"co_occurrence=3, ..."``.
        """
        counts: Dict[int, int] = {}
        for etype in edge_map.values():
            counts[etype] = counts.get(etype, 0) + 1
        if not counts:
            return "no edges"
        return ", ".join(
            f"{edge_type_name(etype)}={counts[etype]}" for etype in sorted(counts)
        )


def build_graph(
    elements: List[Dict[str, Any]],
    label: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper that builds a Content Composition Graph in one call.

    Constructs a throwaway :class:`ContentCompositionGraphBuilder` and delegates
    to :meth:`ContentCompositionGraphBuilder.build`.  Prefer instantiating the
    builder once when processing many samples.

    Parameters
    ----------
    elements : list of dict
        Canonical CHEM element dicts.
    label : int, optional
        Optional ground-truth graph label attached as ``data.y``.
    config_path : str, optional
        Optional override for the policy config path.

    Returns
    -------
    dict
        The same structure returned by
        :meth:`ContentCompositionGraphBuilder.build`.
    """
    return ContentCompositionGraphBuilder(config_path=config_path).build(elements, label)
