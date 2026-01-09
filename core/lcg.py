"""
core/lcg.py

Latent Causal Graph (LCG) implementation.

Responsibilities:
- Represent latent causal variables as nodes
- Represent directed causal influence as edges
- Track confidence per edge
- Support graph comparison and edit distance
- Enable causal consistency evaluation
- Provide serialisable representations for logging and papers
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np


class CausalNode:
    """
    Represents a latent causal variable.
    """

    def __init__(self, name: str):
        self.name: str = name

    def to_dict(self) -> Dict:
        return {"name": self.name}


class CausalEdge:
    """
    Represents a directed causal influence between two nodes.
    """

    def __init__(
        self,
        source: str,
        target: str,
        confidence: float
    ):
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Edge confidence must be between 0 and 1")

        self.source: str = source
        self.target: str = target
        self.confidence: float = confidence

    def to_tuple(self) -> Tuple[str, str]:
        return (self.source, self.target)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "confidence": self.confidence
        }


class LatentCausalGraph:
    """
    Core data structure for latent causal reasoning.
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}

    # ------------------------
    # Node management
    # ------------------------

    def add_node(self, node_name: str) -> None:
        if node_name not in self.nodes:
            self.nodes[node_name] = CausalNode(node_name)

    def has_node(self, node_name: str) -> bool:
        return node_name in self.nodes

    # ------------------------
    # Edge management
    # ------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        confidence: float
    ) -> None:
        if source == target:
            raise ValueError("Self causal edges are not allowed")

        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both source and target nodes must exist")

        edge_key = (source, target)
        self.edges[edge_key] = CausalEdge(source, target, confidence)

    def remove_edge(self, source: str, target: str) -> None:
        edge_key = (source, target)
        if edge_key in self.edges:
            del self.edges[edge_key]

    def get_edge_confidence(self, source: str, target: str) -> Optional[float]:
        edge_key = (source, target)
        if edge_key not in self.edges:
            return None
        return self.edges[edge_key].confidence

    def get_outgoing_edges(self, node_name: str) -> List[CausalEdge]:
        return [
            edge for edge in self.edges.values()
            if edge.source == node_name
        ]

    def get_incoming_edges(self, node_name: str) -> List[CausalEdge]:
        return [
            edge for edge in self.edges.values()
            if edge.target == node_name
        ]

    # ------------------------
    # Graph properties
    # ------------------------

    def adjacency_matrix(self) -> np.ndarray:
        """
        Returns a weighted adjacency matrix ordered by sorted node names.
        """
        node_names = sorted(self.nodes.keys())
        index_map = {name: idx for idx, name in enumerate(node_names)}

        matrix = np.zeros((len(node_names), len(node_names)), dtype=np.float32)

        for (source, target), edge in self.edges.items():
            i = index_map[source]
            j = index_map[target]
            matrix[i, j] = edge.confidence

        return matrix

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    # ------------------------
    # Graph comparison
    # ------------------------

    @staticmethod
    def graph_edit_distance(
        graph_a: LatentCausalGraph,
        graph_b: LatentCausalGraph,
        confidence_threshold: float = 0.0
    ) -> float:
        """
        Computes a normalised graph edit distance between two LCGs.

        Measures:
        - edge additions
        - edge removals
        - confidence changes

        Output is in [0, 1].
        """

        edges_a = {
            edge.to_tuple(): edge.confidence
            for edge in graph_a.edges.values()
            if edge.confidence >= confidence_threshold
        }

        edges_b = {
            edge.to_tuple(): edge.confidence
            for edge in graph_b.edges.values()
            if edge.confidence >= confidence_threshold
        }

        all_edges = set(edges_a.keys()).union(set(edges_b.keys()))

        if len(all_edges) == 0:
            return 0.0

        distance = 0.0

        for edge_key in all_edges:
            conf_a = edges_a.get(edge_key, 0.0)
            conf_b = edges_b.get(edge_key, 0.0)
            distance += abs(conf_a - conf_b)

        normalised_distance = distance / len(all_edges)
        return float(np.clip(normalised_distance, 0.0, 1.0))

    # ------------------------
    # Causal consistency
    # ------------------------

    def causal_consistency(
        self,
        perturbed_graphs: List[LatentCausalGraph],
        confidence_threshold: float = 0.0
    ) -> float:
        """
        Measures how stable this graph is under perturbations.

        Returns a score in [0, 1], where higher is more consistent.
        """
        if len(perturbed_graphs) == 0:
            return 1.0

        distances = [
            LatentCausalGraph.graph_edit_distance(
                self, g, confidence_threshold
            )
            for g in perturbed_graphs
        ]

        mean_distance = float(np.mean(distances))
        consistency_score = 1.0 - mean_distance
        return float(np.clip(consistency_score, 0.0, 1.0))

    # ------------------------
    # Serialisation
    # ------------------------

    def to_dict(self) -> Dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()]
        }

    @staticmethod
    def from_dict(data: Dict) -> LatentCausalGraph:
        graph = LatentCausalGraph()

        for node_data in data.get("nodes", []):
            graph.add_node(node_data["name"])

        for edge_data in data.get("edges", []):
            graph.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                confidence=edge_data["confidence"]
            )

        return graph

    # ------------------------
    # Debugging and display
    # ------------------------

    def summary(self) -> Dict:
        """
        Lightweight summary useful for logging and baselines.
        """
        return {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence
                }
                for edge in self.edges.values()
            ]
        }
