"""
utils/visualisation.py

Strict visualisation utilities for causal graphs and clip level artefacts.

This module is read only by design.
It must never infer causality, modify graphs, compute metrics, or suppress errors.

All visualisation functions operate on fully constructed objects only.
"""

from typing import Optional, Tuple
import math

import matplotlib.pyplot as plt
import networkx as nx

from causal.graph import LatentCausalGraph
from causal.metrics import graph_edit_distance


class VisualisationError(RuntimeError):
    """Raised when visualisation invariants are violated."""
    pass


def _validate_graph(graph: LatentCausalGraph) -> None:
    if not isinstance(graph, LatentCausalGraph):
        raise VisualisationError("Input is not a LatentCausalGraph")

    if graph.nodes is None or graph.edges is None:
        raise VisualisationError("LatentCausalGraph is incomplete")

    for edge in graph.edges:
        if edge.confidence is None:
            raise VisualisationError("Edge confidence must be explicitly defined")

        if not (0.0 <= edge.confidence <= 1.0):
            raise VisualisationError("Edge confidence must be in [0, 1]")


def _to_networkx(graph: LatentCausalGraph) -> nx.DiGraph:
    """
    Convert LatentCausalGraph to a NetworkX DiGraph without losing semantics.
    """
    nx_graph = nx.DiGraph()

    for node in graph.nodes:
        nx_graph.add_node(node.name)

    for edge in graph.edges:
        nx_graph.add_edge(
            edge.source,
            edge.target,
            confidence=edge.confidence
        )

    return nx_graph


def plot_causal_graph(
    graph: LatentCausalGraph,
    title: Optional[str] = None,
    layout_seed: int = 42
) -> None:
    """
    Visualise a LatentCausalGraph with edge confidence encoded.

    Edge thickness and transparency scale with confidence.
    """
    _validate_graph(graph)

    nx_graph = _to_networkx(graph)

    pos = nx.spring_layout(nx_graph, seed=layout_seed)

    confidences = []
    for _, _, data in nx_graph.edges(data=True):
        confidences.append(data["confidence"])

    if len(confidences) == 0:
        raise VisualisationError("Cannot visualise graph with no edges")

    widths = [1.0 + 4.0 * c for c in confidences]
    alphas = [0.3 + 0.7 * c for c in confidences]

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_size=1500,
        node_color="#E6E6E6",
        edgecolors="#000000"
    )

    for (edge, width, alpha) in zip(nx_graph.edges(), widths, alphas):
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edgelist=[edge],
            width=width,
            alpha=alpha,
            arrows=True,
            arrowstyle="-|>"
        )

    nx.draw_networkx_labels(
        nx_graph,
        pos,
        font_size=10
    )

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_graph_comparison(
    graph_a: LatentCausalGraph,
    graph_b: LatentCausalGraph,
    title: Optional[str] = None
) -> None:
    """
    Visualise structural difference between two LatentCausalGraphs.

    This function only visualises.
    It does not interpret, score, or judge quality.
    """
    _validate_graph(graph_a)
    _validate_graph(graph_b)

    distance = graph_edit_distance(graph_a, graph_b)

    nx_a = _to_networkx(graph_a)
    nx_b = _to_networkx(graph_b)

    pos = nx.spring_layout(nx.compose(nx_a, nx_b), seed=42)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    nx.draw(
        nx_a,
        pos,
        with_labels=True,
        node_color="#DDEEFF",
        edge_color="#3366CC"
    )
    plt.title("Graph A")

    plt.subplot(1, 2, 2)
    nx.draw(
        nx_b,
        pos,
        with_labels=True,
        node_color="#FFEEDD",
        edge_color="#CC6633"
    )
    plt.title("Graph B")

    if title is not None:
        plt.suptitle(f"{title} | graph_edit_distance = {distance}")

    plt.tight_layout()
    plt.show()


def sanity_check_visualisation(graph: LatentCausalGraph) -> None:
    """
    Fail fast utility used in tests to confirm a graph is visualisable.
    """
    _validate_graph(graph)

    for edge in graph.edges:
        if math.isnan(edge.confidence) or math.isinf(edge.confidence):
            raise VisualisationError("Invalid numerical value in edge confidence")
