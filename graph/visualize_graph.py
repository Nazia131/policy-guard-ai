"""Visualisation of the CHEM Content Composition Graph.

Renders the NetworkX ``DiGraph`` produced by
:class:`graph.graph_builder.ContentCompositionGraphBuilder` into a labelled
matplotlib figure so humans (and reviewers / explainability reports) can see how
individual modality elements compose into a potentially harmful whole.

Design choices:

* **Node colour** encodes *modality* (text / image / audio / video / action).
* **Node shape emphasis** -- harmful nodes get a bold red border.
* **Edge colour + line style** encode the *edge type* (co_occurrence,
  semantic_reference, temporal_sequence, target_pointing, harmful_pattern).
* A two-part legend explains both modality colours and edge-type styles.

Public API
----------
visualize_graph(graph_dict, out_path=None) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx

from graph.edge_types import EDGE_TYPES, edge_type_name


# Colour per modality.  Purely presentational (no harm semantics encoded).
_MODALITY_COLOURS: Dict[str, str] = {
    "text": "#4C78A8",   # blue
    "image": "#59A14F",  # green
    "audio": "#B279A2",  # purple
    "video": "#F58518",  # orange
    "action": "#E45756", # red-ish
}
_DEFAULT_MODALITY_COLOUR = "#BAB0AC"  # grey fallback for unknown modality

# Colour + line style per edge type name.  Keys match ``policy_config.json``.
_EDGE_STYLES: Dict[str, Dict[str, Any]] = {
    "co_occurrence": {"color": "#9C9C9C", "style": "solid"},
    "semantic_reference": {"color": "#4C78A8", "style": "dashed"},
    "temporal_sequence": {"color": "#54A24B", "style": "dotted"},
    "target_pointing": {"color": "#F58518", "style": "dashdot"},
    "harmful_pattern": {"color": "#D62728", "style": "solid"},
}
_DEFAULT_EDGE_STYLE = {"color": "#000000", "style": "solid"}


def _log(message: str) -> None:
    """Print a namespaced progress log line for the visualiser.

    Parameters
    ----------
    message : str
        Message to print (prefixed with ``[CHEM][graph_builder]``).

    Returns
    -------
    None
    """
    print(f"[CHEM][graph_builder] {message}")


def visualize_graph(
    graph_dict: Dict[str, Any],
    out_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Draw a Content Composition Graph with matplotlib.

    Parameters
    ----------
    graph_dict : dict
        The dict returned by
        :meth:`graph.graph_builder.ContentCompositionGraphBuilder.build`.  Must
        contain a ``"networkx"`` :class:`networkx.DiGraph`; if it is missing the
        graph is reconstructed from the ``"nodes"`` / ``"edges"`` lists.
    out_path : str, optional
        If given, the figure is saved to this path (PNG/SVG/etc. inferred from
        the extension) and the figure is closed.  If ``None`` the figure is
        shown interactively via ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (also saved/shown as a side effect).
    """
    graph = _resolve_graph(graph_dict)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    _log(f"Visualising graph with {n_nodes} node(s) and {n_edges} edge(s)")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_axis_off()
    ax.set_title("CHEM Content Composition Graph", fontsize=14, fontweight="bold")

    # Empty graph: render an informative placeholder rather than crashing.
    if n_nodes == 0:
        ax.text(
            0.5, 0.5, "empty graph (no elements)",
            ha="center", va="center", fontsize=12, color="#888888",
            transform=ax.transAxes,
        )
        return _finalise(fig, out_path)

    # Deterministic layout so repeated renders are stable.
    pos = nx.spring_layout(graph, seed=42, k=0.9)

    _draw_nodes(graph, pos, ax)
    _draw_edges(graph, pos, ax)
    _draw_labels(graph, pos, ax)
    _add_legend(graph, ax)

    return _finalise(fig, out_path)


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #
def _resolve_graph(graph_dict: Dict[str, Any]) -> "nx.DiGraph":
    """Return a NetworkX ``DiGraph`` for the given graph dict.

    Uses the pre-built ``"networkx"`` graph when present, otherwise rebuilds one
    from the JSON-serialisable ``"nodes"`` / ``"edges"`` lists.

    Parameters
    ----------
    graph_dict : dict
        A CHEM graph dict.

    Returns
    -------
    networkx.DiGraph
        The directed graph to draw.
    """
    graph = graph_dict.get("networkx")
    if isinstance(graph, nx.DiGraph):
        return graph

    rebuilt = nx.DiGraph()
    for node in graph_dict.get("nodes", []):
        idx = node.get("index")
        rebuilt.add_node(idx, **node)
    for edge in graph_dict.get("edges", []):
        etype = edge.get("edge_type", 0)
        rebuilt.add_edge(
            edge.get("source"),
            edge.get("target"),
            edge_type=etype,
            edge_type_name=edge.get("edge_type_name", edge_type_name(etype)),
        )
    return rebuilt


def _draw_nodes(graph: "nx.DiGraph", pos: Dict[Any, Any], ax) -> None:
    """Draw nodes coloured by modality, with harmful nodes outlined in red.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph being drawn.
    pos : dict
        Node -> (x, y) layout positions.
    ax : matplotlib.axes.Axes
        Axis to draw on.

    Returns
    -------
    None
    """
    node_list = list(graph.nodes())
    colours = [
        _MODALITY_COLOURS.get(graph.nodes[n].get("modality"), _DEFAULT_MODALITY_COLOUR)
        for n in node_list
    ]
    # Harmful nodes get a thick red border; benign nodes a thin grey one.
    edge_colours = [
        "#D62728" if graph.nodes[n].get("harmful_signal") else "#444444"
        for n in node_list
    ]
    line_widths = [
        2.5 if graph.nodes[n].get("harmful_signal") else 1.0 for n in node_list
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list,
        node_color=colours,
        edgecolors=edge_colours,
        linewidths=line_widths,
        node_size=1300,
        ax=ax,
    )


def _draw_edges(graph: "nx.DiGraph", pos: Dict[Any, Any], ax) -> None:
    """Draw edges grouped by edge type, each with its own colour and style.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph being drawn.
    pos : dict
        Node -> (x, y) layout positions.
    ax : matplotlib.axes.Axes
        Axis to draw on.

    Returns
    -------
    None
    """
    # Group edges by type name so each style is drawn in a single call.
    grouped: Dict[str, list] = {}
    for u, v, attrs in graph.edges(data=True):
        name = attrs.get("edge_type_name") or edge_type_name(attrs.get("edge_type", 0))
        grouped.setdefault(name, []).append((u, v))

    for name, edges in grouped.items():
        style = _EDGE_STYLES.get(name, _DEFAULT_EDGE_STYLE)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges,
            edge_color=style["color"],
            style=style["style"],
            width=2.2 if name == "harmful_pattern" else 1.5,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.08",
            node_size=1300,
            ax=ax,
        )


def _draw_labels(graph: "nx.DiGraph", pos: Dict[Any, Any], ax) -> None:
    """Draw short text labels (element id / label) on each node.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph being drawn.
    pos : dict
        Node -> (x, y) layout positions.
    ax : matplotlib.axes.Axes
        Axis to draw on.

    Returns
    -------
    None
    """
    labels = {}
    for n in graph.nodes():
        meta = graph.nodes[n]
        labels[n] = str(meta.get("id") or meta.get("label") or n)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)


def _add_legend(graph: "nx.DiGraph", ax) -> None:
    """Add a combined modality + edge-type legend to the axis.

    Only the modalities / edge types actually present in the graph are listed to
    keep the legend compact.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph being drawn.
    ax : matplotlib.axes.Axes
        Axis to draw on.

    Returns
    -------
    None
    """
    present_modalities = {
        graph.nodes[n].get("modality") for n in graph.nodes()
    }
    present_edge_types = {
        attrs.get("edge_type_name") or edge_type_name(attrs.get("edge_type", 0))
        for _, _, attrs in graph.edges(data=True)
    }

    handles = []
    # Modality colour swatches.
    for modality, colour in _MODALITY_COLOURS.items():
        if modality in present_modalities:
            handles.append(Patch(facecolor=colour, edgecolor="#444444", label=f"node: {modality}"))
    # Edge-type line samples (ordered by canonical id).
    for name in sorted(present_edge_types, key=lambda nm: EDGE_TYPES.get(nm, 99)):
        style = _EDGE_STYLES.get(name, _DEFAULT_EDGE_STYLE)
        handles.append(
            Line2D([0], [0], color=style["color"], linestyle=style["style"],
                   linewidth=2, label=f"edge: {name}")
        )

    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)


def _finalise(fig, out_path: Optional[str]) -> "matplotlib.figure.Figure":
    """Save or show the figure and return it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalise.
    out_path : str, optional
        Destination path; if ``None`` the figure is shown interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure object.
    """
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        _log(f"Saved visualisation to {out_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig
