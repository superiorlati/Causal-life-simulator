"""
utils/logging.py

Centralised experiment and baseline logging.

Responsibilities:
- Deterministic experiment logging
- Baseline storage and versioning
- Metric aggregation
- LatentCausalGraph serialisation
- CSV and JSON outputs for papers
- Reproducibility guarantees

This is the ONLY logging interface used by the system.
"""

from __future__ import annotations

import os
import json
import csv
import time
from typing import Dict, List, Optional, Any

import numpy as np

from core.lcg import LatentCausalGraph


class ExperimentLogger:
    """
    Unified logger for experiments, baselines, and evaluations.
    """

    def __init__(
        self,
        log_root: str = "logs",
        experiment_name: Optional[str] = None
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.experiment_name: str = (
            experiment_name if experiment_name is not None else f"experiment_{timestamp}"
        )

        self.log_root: str = log_root
        self.experiment_dir: str = os.path.join(log_root, self.experiment_name)

        self.baseline_dir: str = os.path.join(self.experiment_dir, "baselines")
        self.metrics_dir: str = os.path.join(self.experiment_dir, "metrics")
        self.graphs_dir: str = os.path.join(self.experiment_dir, "graphs")

        self._create_directories()

    def _create_directories(self) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.baseline_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

    # -------------------------------------------------
    # Baseline logging
    # -------------------------------------------------

    def log_baseline(
        self,
        baseline_name: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Logs a frozen baseline snapshot.
        """
        path = os.path.join(self.baseline_dir, f"{baseline_name}.json")
        self._write_json(path, data)

    # -------------------------------------------------
    # Metric logging
    # -------------------------------------------------

    def log_metrics(
        self,
        metrics_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Logs scalar metrics for later aggregation.
        """
        path = os.path.join(self.metrics_dir, f"{metrics_name}.json")
        self._write_json(path, metrics)

    def log_metrics_csv(
        self,
        metrics_name: str,
        metrics_list: List[Dict[str, float]]
    ) -> None:
        """
        Logs repeated metrics to CSV for plots and tables.
        """
        if len(metrics_list) == 0:
            return

        path = os.path.join(self.metrics_dir, f"{metrics_name}.csv")
        fieldnames = list(metrics_list[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_list:
                writer.writerow(row)

    # -------------------------------------------------
    # Graph logging
    # -------------------------------------------------

    def log_graph(
        self,
        graph_name: str,
        graph: LatentCausalGraph
    ) -> None:
        """
        Logs a LatentCausalGraph to disk in serialised form.
        """
        path = os.path.join(self.graphs_dir, f"{graph_name}.json")
        self._write_json(path, graph.to_dict())

    def log_graph_comparison(
        self,
        comparison_name: str,
        graph_a: LatentCausalGraph,
        graph_b: LatentCausalGraph,
        confidence_threshold: float = 0.0
    ) -> None:
        """
        Logs structural comparison between two graphs.
        """
        distance = LatentCausalGraph.graph_edit_distance(
            graph_a, graph_b, confidence_threshold
        )

        data = {
            "graph_edit_distance": distance,
            "confidence_threshold": confidence_threshold,
            "graph_a": graph_a.summary(),
            "graph_b": graph_b.summary()
        }

        path = os.path.join(self.graphs_dir, f"{comparison_name}.json")
        self._write_json(path, data)

    # -------------------------------------------------
    # Utility logging
    # -------------------------------------------------

    def log_array_statistics(
        self,
        name: str,
        values: List[float]
    ) -> None:
        """
        Logs summary statistics for numeric arrays.
        """
        if len(values) == 0:
            return

        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

        path = os.path.join(self.metrics_dir, f"{name}_stats.json")
        self._write_json(path, stats)

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
