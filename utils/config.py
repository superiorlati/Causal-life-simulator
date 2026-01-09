"""
utils/config.py

Central configuration management for the entire system.

Responsibilities:
- Single source of truth for all hyperparameters
- Explicit phase level configuration
- Reproducibility and experiment freezing
- Safe defaults suitable for paper submission
- No dynamic mutation at runtime

This file contains NO logic related to learning or inference.
It only defines configuration structures and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


# -------------------------------------------------
# Dataset configuration
# -------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for DatasetLoader and VideoClip handling.
    """
    dataset_root: str = "data/videos"
    max_clips: int = 10
    resize_height: int = 224
    resize_width: int = 224
    grayscale: bool = False
    fps_override: float | None = None


# -------------------------------------------------
# Perception phase configuration
# -------------------------------------------------

@dataclass(frozen=True)
class PerceptionConfig:
    """
    Configuration for perception_agent.py
    """
    optical_flow_method: str = "farneback"
    flow_pyramid_scale: float = 0.5
    flow_levels: int = 3
    occlusion_threshold: float = 0.4
    uncertainty_smoothing: float = 0.1


# -------------------------------------------------
# Causal graph configuration
# -------------------------------------------------

@dataclass(frozen=True)
class CausalGraphConfig:
    """
    Configuration for LatentCausalGraph construction and comparison.
    """
    min_edge_confidence: float = 0.2
    max_nodes: int = 64
    confidence_decay: float = 0.95
    perturbation_noise: float = 0.01


# -------------------------------------------------
# Temporal mechanics configuration
# -------------------------------------------------

@dataclass(frozen=True)
class TemporalConfig:
    """
    Configuration for temporal rewind and causal locking.
    """
    max_snapshots: int = 500
    allow_partial_rewind: bool = True
    frozen_variable_penalty: float = 1.0


# -------------------------------------------------
# Future branching configuration
# -------------------------------------------------

@dataclass(frozen=True)
class FutureBranchingConfig:
    """
    Configuration for future branching agent.
    """
    max_branches: int = 12
    min_graph_edit_distance: float = 1.0
    causal_diversity_weight: float = 1.0
    reject_low_confidence_futures: bool = True


# -------------------------------------------------
# Intervention search configuration
# -------------------------------------------------

@dataclass(frozen=True)
class InterventionConfig:
    """
    Configuration for intervention search.
    """
    max_interventions: int = 20
    intervention_strength: float = 0.5
    outcome_flip_threshold: float = 0.3


# -------------------------------------------------
# Self critique and verification configuration
# -------------------------------------------------

@dataclass(frozen=True)
class SelfCritiqueConfig:
    """
    Configuration for sceptic and verification agents.
    """
    edge_removal_fraction: float = 0.2
    max_alternative_explanations: int = 5
    verification_reruns: int = 3
    correction_tolerance: float = 0.05


# -------------------------------------------------
# Experiment level configuration
# -------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Master configuration object passed through the entire system.
    """
    dataset: DatasetConfig = DatasetConfig()
    perception: PerceptionConfig = PerceptionConfig()
    causal_graph: CausalGraphConfig = CausalGraphConfig()
    temporal: TemporalConfig = TemporalConfig()
    future_branching: FutureBranchingConfig = FutureBranchingConfig()
    intervention: InterventionConfig = InterventionConfig()
    self_critique: SelfCritiqueConfig = SelfCritiqueConfig()

    random_seed: int = 42
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialises the entire configuration tree.
        """
        return {
            "dataset": asdict(self.dataset),
            "perception": asdict(self.perception),
            "causal_graph": asdict(self.causal_graph),
            "temporal": asdict(self.temporal),
            "future_branching": asdict(self.future_branching),
            "intervention": asdict(self.intervention),
            "self_critique": asdict(self.self_critique),
            "random_seed": self.random_seed,
            "deterministic": self.deterministic,
        }
