"""Frequent connected subgraph mining for the CHEM harm-pattern library.

This module discovers *compositional* harm patterns: small connected,
node-labelled subgraphs that appear frequently across content-composition
graphs labelled *harmful* while remaining essentially absent from graphs
labelled *safe*.  These mined subgraphs become the reusable "harm patterns"
that the downstream scorer matches new content against.

The algorithm is a pragmatic, NetworkX-based approximation of gSpan-style
frequent-subgraph mining:

1. Normalise every input graph, attaching a compact node label
   (``label``/``type``) and edge label (``edge_type``) used for matching.
2. Enumerate all connected node-labelled subgraphs up to ``max_nodes`` nodes
   inside the harmful graphs.
3. Canonicalise each subgraph (Weisfeiler-Lehman graph hash over the label
   attributes) so isomorphic occurrences collapse to a single candidate.
4. Count *support* for each candidate via subgraph (mono)morphism matching
   with node/edge label constraints, separately over harmful and safe graphs.
5. Keep candidates that are frequent in harmful graphs
   (``support >= min_support``) and rare in safe graphs
   (``safe_support < SAFE_SUPPORT_CEILING``).

No harm semantics are hardcoded here; labels come from the graphs (which in
turn come from ``policy_config.json``).  This file only performs structural
mining.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms import isomorphism as iso


def _edge_type_name(value: int) -> str:
    """Resolve an integer edge-type id to its canonical name.

    The ``graph.edge_types`` helper is imported lazily so the miner does not
    hard-depend on that sibling module at import time; if it is unavailable the
    raw integer is stringified instead.

    Parameters
    ----------
    value : int
        The integer edge-type id.

    Returns
    -------
    str
        The canonical edge-type name, or the stringified integer on failure.
    """
    try:
        from graph.edge_types import edge_type_name

        return edge_type_name(int(value))
    except Exception:
        return str(value)

# Maximum fraction of *safe* graphs a pattern may occur in and still be
# considered harm-specific.  This is a mining hygiene constant (a
# discriminativeness filter), not a harm-governing policy threshold.
SAFE_SUPPORT_CEILING: float = 0.02

# Safety cap on how many distinct connected subgraphs we enumerate per graph,
# so a pathologically dense graph cannot blow up enumeration time.
_MAX_SUBSETS_PER_GRAPH: int = 20000

# Internal attribute keys used purely for label-aware isomorphism matching.
_NODE_LABEL_KEY = "__chem_node_label"
_EDGE_LABEL_KEY = "__chem_edge_label"


class SubgraphMiner:
    """Mine frequent, harm-discriminative connected subgraphs.

    Parameters
    ----------
    max_nodes : int, optional
        Maximum number of nodes in a mined subgraph pattern (default ``4``).
        Larger values discover richer compositions at higher compute cost.
    safe_support_ceiling : float, optional
        A pattern is discarded if its support fraction among *safe* graphs is
        greater than or equal to this value (default :data:`SAFE_SUPPORT_CEILING`).
    verbose : bool, optional
        When ``True`` (default) print ``[CHEM][subgraph_miner]`` progress logs.
    """

    def __init__(
        self,
        max_nodes: int = 4,
        safe_support_ceiling: float = SAFE_SUPPORT_CEILING,
        verbose: bool = True,
    ) -> None:
        """Initialise the miner and remember its enumeration bounds."""
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        self.max_nodes = int(max_nodes)
        self.safe_support_ceiling = float(safe_support_ceiling)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def mine(
        self,
        harmful_graphs: List[nx.Graph],
        safe_graphs: List[nx.Graph],
        min_support: float,
    ) -> List[Dict[str, Any]]:
        """Mine frequent, harm-discriminative subgraph patterns.

        Parameters
        ----------
        harmful_graphs : list of networkx.Graph
            Content-composition graphs labelled harmful.  Each graph may carry
            a graph-level ``content_id`` attribute (``G.graph['content_id']``)
            used to record a representative example for each pattern.
        safe_graphs : list of networkx.Graph
            Content-composition graphs labelled safe, used to compute the
            discriminative ``safe_support`` of each candidate.
        min_support : float
            Minimum support fraction (``0..1``) a pattern must reach among the
            harmful graphs to be retained.

        Returns
        -------
        list of dict
            One dict per retained pattern with keys::

                {
                    "pattern_id": str,
                    "graph": dict,               # nx.node_link_data JSON
                    "support": float,            # fraction of harmful graphs
                    "safe_support": float,       # fraction of safe graphs
                    "size": int,                 # number of nodes
                    "example_content_id": str,   # a harmful graph exhibiting it
                }

            An empty list is returned for empty inputs.
        """
        harmful_graphs = list(harmful_graphs or [])
        safe_graphs = list(safe_graphs or [])

        if not harmful_graphs:
            if self.verbose:
                print(
                    "[CHEM][subgraph_miner] No harmful graphs supplied; "
                    "nothing to mine."
                )
            return []

        if self.verbose:
            print(
                f"[CHEM][subgraph_miner] Mining from {len(harmful_graphs)} "
                f"harmful and {len(safe_graphs)} safe graphs "
                f"(max_nodes={self.max_nodes}, min_support={min_support})."
            )

        # Normalise every graph once: attach compact match labels.
        norm_harmful = [self._normalise(g) for g in harmful_graphs]
        norm_safe = [self._normalise(g) for g in safe_graphs]

        # ---- Step 1: enumerate + canonicalise candidate subgraphs -------- #
        # canonical_key -> {"template": nx graph, "example_content_id": str}
        candidates: Dict[str, Dict[str, Any]] = {}
        for gidx, g in enumerate(norm_harmful):
            content_id = str(g.graph.get("content_id", f"harmful_{gidx}"))
            n_new = self._collect_candidates(g, content_id, candidates)
            if self.verbose:
                print(
                    f"[CHEM][subgraph_miner]   graph {gidx + 1}/"
                    f"{len(norm_harmful)}: {g.number_of_nodes()} nodes -> "
                    f"{n_new} new candidate patterns "
                    f"(total {len(candidates)})."
                )

        if self.verbose:
            print(
                f"[CHEM][subgraph_miner] Enumerated {len(candidates)} distinct "
                f"candidate patterns; scoring support."
            )

        # ---- Step 2: score support and filter ---------------------------- #
        n_harmful = len(norm_harmful)
        n_safe = len(norm_safe)
        kept: List[Dict[str, Any]] = []

        for cidx, (key, info) in enumerate(candidates.items()):
            template = info["template"]

            harmful_hits = sum(
                1 for g in norm_harmful if self._contains(g, template)
            )
            support = harmful_hits / n_harmful if n_harmful else 0.0
            if support < min_support:
                continue

            safe_hits = (
                sum(1 for g in norm_safe if self._contains(g, template))
                if n_safe
                else 0
            )
            safe_support = safe_hits / n_safe if n_safe else 0.0
            if safe_support >= self.safe_support_ceiling:
                continue

            clean = self._strip_internal(template)
            kept.append(
                {
                    "pattern_id": f"pattern_{len(kept):04d}",
                    "graph": _node_link_data(clean),
                    "support": round(float(support), 4),
                    "safe_support": round(float(safe_support), 4),
                    "size": int(clean.number_of_nodes()),
                    "example_content_id": info["example_content_id"],
                }
            )

            if self.verbose and (cidx + 1) % 50 == 0:
                print(
                    f"[CHEM][subgraph_miner]   scored {cidx + 1}/"
                    f"{len(candidates)} candidates; {len(kept)} kept so far."
                )

        # Rank most frequent / most discriminative patterns first.
        kept.sort(key=lambda p: (p["support"], -p["safe_support"]), reverse=True)
        for i, pat in enumerate(kept):
            pat["pattern_id"] = f"pattern_{i:04d}"

        if self.verbose:
            print(
                f"[CHEM][subgraph_miner] Done. Kept {len(kept)} harm-"
                f"discriminative patterns."
            )
        return kept

    # ------------------------------------------------------------------ #
    # Candidate enumeration
    # ------------------------------------------------------------------ #
    def _collect_candidates(
        self,
        g: nx.Graph,
        content_id: str,
        candidates: Dict[str, Dict[str, Any]],
    ) -> int:
        """Enumerate connected subgraphs of ``g`` and register new candidates.

        Parameters
        ----------
        g : networkx.Graph
            A normalised graph (already carrying internal label attributes).
        content_id : str
            Identifier recorded as the example for any newly discovered pattern.
        candidates : dict
            Accumulator mapping canonical key -> candidate info, mutated in place.

        Returns
        -------
        int
            The number of *newly registered* candidate patterns from this graph.
        """
        added = 0
        for node_set in self._connected_node_subsets(g):
            template = g.subgraph(node_set).copy()
            key = self._canonical_key(template)
            if key not in candidates:
                candidates[key] = {
                    "template": template,
                    "example_content_id": content_id,
                }
                added += 1
        return added

    def _connected_node_subsets(self, g: nx.Graph) -> Iterable[Set[Any]]:
        """Yield every connected node subset of size ``1..max_nodes``.

        The graph is treated as undirected for connectivity purposes so that a
        directed edge still joins its endpoints into one subgraph.

        Parameters
        ----------
        g : networkx.Graph
            Graph whose connected node subsets are enumerated.

        Yields
        ------
        set
            A set of node ids forming a connected induced subgraph.
        """
        undirected = g.to_undirected(as_view=True)
        seen: Set[frozenset] = set()
        emitted = 0

        for start in undirected.nodes():
            # Grow connected subsets starting from ``start`` via DFS.  To avoid
            # duplicate work we only ever extend with neighbours whose id is
            # "greater" than the start under a stable ordering is unnecessary
            # here because ``seen`` deduplicates frozensets directly.
            stack: List[Tuple[frozenset, frozenset]] = [
                (frozenset({start}), frozenset(undirected.neighbors(start)) - {start})
            ]
            while stack:
                current, frontier = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                yield set(current)
                emitted += 1
                if emitted >= _MAX_SUBSETS_PER_GRAPH:
                    return
                if len(current) >= self.max_nodes:
                    continue
                for nxt in frontier:
                    new_current = current | {nxt}
                    if new_current in seen:
                        continue
                    new_frontier = (
                        frontier | frozenset(undirected.neighbors(nxt))
                    ) - new_current
                    stack.append((new_current, new_frontier))

    # ------------------------------------------------------------------ #
    # Canonicalisation and matching helpers
    # ------------------------------------------------------------------ #
    def _canonical_key(self, template: nx.Graph) -> str:
        """Return a canonical key that is identical for isomorphic patterns.

        Uses the Weisfeiler-Lehman graph hash over the internal node/edge label
        attributes so two label-preserving isomorphic subgraphs collapse to one
        candidate.  Falls back to a deterministic label-multiset signature when
        the installed NetworkX lacks the WL hash helper.

        Parameters
        ----------
        template : networkx.Graph
            A normalised candidate subgraph.

        Returns
        -------
        str
            The canonical key string.
        """
        # WL hash is defined on undirected graphs; direction is approximated by
        # the label-multiset fallback contribution appended below.
        undirected = nx.Graph(template.to_undirected())
        try:
            wl = nx.weisfeiler_lehman_graph_hash(
                undirected,
                node_attr=_NODE_LABEL_KEY,
                edge_attr=_EDGE_LABEL_KEY,
                iterations=self.max_nodes,
            )
        except Exception:  # pragma: no cover - depends on nx version
            wl = ""
        return f"{wl}|{self._label_signature(template)}"

    @staticmethod
    def _label_signature(template: nx.Graph) -> str:
        """Return a deterministic multiset signature of node/edge labels.

        Parameters
        ----------
        template : networkx.Graph
            A normalised candidate subgraph.

        Returns
        -------
        str
            A signature combining sorted node labels and directed labelled
            edge triples.  Two isomorphic labelled graphs share this signature.
        """
        node_labels = sorted(
            str(data.get(_NODE_LABEL_KEY, "")) for _, data in template.nodes(data=True)
        )
        edge_triples = sorted(
            "{}->{}:{}".format(
                template.nodes[u].get(_NODE_LABEL_KEY, ""),
                template.nodes[v].get(_NODE_LABEL_KEY, ""),
                data.get(_EDGE_LABEL_KEY, ""),
            )
            for u, v, data in template.edges(data=True)
        )
        return "N[" + ",".join(node_labels) + "]E[" + ",".join(edge_triples) + "]"

    def _contains(self, big: nx.Graph, pattern: nx.Graph) -> bool:
        """Return ``True`` if ``big`` contains a match of ``pattern``.

        Matching is label-aware subgraph monomorphism: a subgraph of ``big``
        must reproduce ``pattern``'s structure with identical node labels
        (``label``/``type``) and edge labels (``edge_type``).  Monomorphism
        (rather than induced isomorphism) is used so extra edges in ``big`` do
        not prevent a pattern occurrence from being recognised.

        Parameters
        ----------
        big : networkx.Graph
            The (normalised) host graph being searched.
        pattern : networkx.Graph
            The (normalised) pattern being sought.

        Returns
        -------
        bool
            Whether the pattern occurs in the host graph.
        """
        node_match = iso.categorical_node_match(_NODE_LABEL_KEY, None)
        edge_match = iso.categorical_edge_match(_EDGE_LABEL_KEY, None)
        try:
            if big.is_directed() and pattern.is_directed():
                matcher = iso.DiGraphMatcher(
                    big, pattern, node_match=node_match, edge_match=edge_match
                )
            else:
                matcher = iso.GraphMatcher(
                    nx.Graph(big),
                    nx.Graph(pattern),
                    node_match=node_match,
                    edge_match=edge_match,
                )
            return matcher.subgraph_is_monomorphic()
        except Exception:  # pragma: no cover - defensive
            return False

    # ------------------------------------------------------------------ #
    # Normalisation helpers
    # ------------------------------------------------------------------ #
    def _normalise(self, g: nx.Graph) -> nx.Graph:
        """Return a copy of ``g`` with internal match-label attributes attached.

        The original node/edge attributes are preserved; two extra keys are
        added:

        * :data:`_NODE_LABEL_KEY` — the node's ``label`` (fallback ``type``).
        * :data:`_EDGE_LABEL_KEY` — the edge's ``edge_type`` normalised to its
          canonical string name.

        Parameters
        ----------
        g : networkx.Graph
            A content-composition graph.

        Returns
        -------
        networkx.Graph
            A normalised copy safe to mutate and match against.
        """
        out = g.copy()
        for _, data in out.nodes(data=True):
            data[_NODE_LABEL_KEY] = _node_label(data)
        for _, _, data in out.edges(data=True):
            data[_EDGE_LABEL_KEY] = _edge_label(data)
        return out

    @staticmethod
    def _strip_internal(template: nx.Graph) -> nx.Graph:
        """Return a copy of ``template`` without the internal match-label keys.

        Parameters
        ----------
        template : networkx.Graph
            A normalised pattern subgraph.

        Returns
        -------
        networkx.Graph
            A clean copy exposing only the original node/edge attributes, ready
            for JSON serialisation and rule-based tagging.
        """
        clean = template.copy()
        for _, data in clean.nodes(data=True):
            data.pop(_NODE_LABEL_KEY, None)
        for _, _, data in clean.edges(data=True):
            data.pop(_EDGE_LABEL_KEY, None)
        return clean


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def _node_label(attrs: Dict[str, Any]) -> str:
    """Return the compact matching label for a node.

    Parameters
    ----------
    attrs : dict
        Node attribute dictionary.

    Returns
    -------
    str
        The node ``label`` if present, else its ``type``, else ``"node"``.
    """
    for key in ("label", "type", "modality"):
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return "node"


def _edge_label(attrs: Dict[str, Any]) -> str:
    """Return the canonical string name of an edge's ``edge_type``.

    Accepts either an integer edge-type id or an already-named string and maps
    it to the canonical name from the policy configuration.

    Parameters
    ----------
    attrs : dict
        Edge attribute dictionary.

    Returns
    -------
    str
        The canonical edge-type name, or ``"unknown"`` if unavailable.
    """
    value = attrs.get("edge_type", attrs.get("type"))
    if value is None:
        return "unknown"
    if isinstance(value, bool):  # guard: bools are ints in Python
        return str(value)
    if isinstance(value, (int,)):
        return _edge_type_name(int(value))
    return str(value)


def _node_link_data(graph: nx.Graph) -> Dict[str, Any]:
    """Serialise a NetworkX graph to node-link JSON in a version-safe way.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to serialise.

    Returns
    -------
    dict
        A ``networkx.node_link_data`` dictionary.
    """
    try:
        # Newer NetworkX emits a FutureWarning unless ``edges`` is specified.
        return nx.node_link_data(graph, edges="links")
    except TypeError:  # pragma: no cover - older NetworkX signature
        return nx.node_link_data(graph)
