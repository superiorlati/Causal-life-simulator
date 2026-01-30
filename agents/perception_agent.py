import cv2
import numpy as np
from typing import Dict, List, Any

from dataset.loader import VideoClip
from utils.logging import ExperimentLogger


class PerceptionAgent:
    """
    Phase 1 agent.

    Responsibilities:
    - Extract perceptual latent variables only
    - No causality
    - No prediction
    - No futures
    - No outcomes

    Outputs are purely descriptive and uncertain.
    """

    def __init__(self, logger: ExperimentLogger):
        self.logger = logger

    def run(self, clip: VideoClip) -> Dict[str, Any]:
        """
        Run perception on a single VideoClip.

        Returns a dictionary of perceptual variables and uncertainty estimates.

        This function must never:
        - infer causality
        - infer outcomes
        - generate futures
        """

        if clip.held_out:
            raise ValueError("PerceptionAgent must not run on held-out clips.")

        if len(clip.frames) < 2:
            raise ValueError("Clip must contain at least two frames.")

        frames_gray = self._convert_to_grayscale(clip.frames)

        optical_flows = self._compute_optical_flow(frames_gray)
        motion_vectors = self._compute_motion_vectors(optical_flows)
        occlusion_map = self._estimate_occlusion(optical_flows)
        uncertainty = self._estimate_uncertainty(optical_flows, occlusion_map)

        results = {
            "clip_fps": clip.fps,
            "num_frames": len(clip.frames),
            "optical_flow": optical_flows,
            "motion_vectors": motion_vectors,
            "occlusion_map": occlusion_map,
            "uncertainty": uncertainty
        }

        self._sanity_check(results)

        return results

    def _convert_to_grayscale(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        gray_frames = []
        for idx, frame in enumerate(frames):
            if frame is None:
                raise ValueError(f"Frame {idx} is None.")
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            gray_frames.append(gray)
        return gray_frames

    def _compute_optical_flow(self, gray_frames: List[np.ndarray]) -> List[np.ndarray]:
        flows = []
        for i in range(len(gray_frames) - 1):
            prev = gray_frames[i]
            nxt = gray_frames[i + 1]

            flow = cv2.calcOpticalFlowFarneback(
                prev,
                nxt,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            if flow is None:
                raise RuntimeError(f"Optical flow failed at frame {i}.")

            flows.append(flow)

        return flows

    def _compute_motion_vectors(self, flows: List[np.ndarray]) -> np.ndarray:
        """
        Compute relative motion vectors as mean magnitude and direction per frame.
        """
        magnitudes = []
        angles = []

        for flow in flows:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(mag))
            angles.append(np.mean(ang))

        return np.stack([magnitudes, angles], axis=1)

    def _estimate_occlusion(self, flows: List[np.ndarray]) -> np.ndarray:
        """
        Simple occlusion proxy based on flow divergence.
        """
        occlusion_scores = []

        for flow in flows:
            dx = flow[..., 0]
            dy = flow[..., 1]
            divergence = np.abs(np.gradient(dx)[0] + np.gradient(dy)[1])
            occlusion_scores.append(np.mean(divergence))

        return np.array(occlusion_scores)

    def _estimate_uncertainty(
        self,
        flows: List[np.ndarray],
        occlusion_map: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate uncertainty from flow variance and occlusion instability.
        """
        flow_variances = []
        for flow in flows:
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_variances.append(np.var(mag))

        uncertainty = {
            "mean_flow_variance": float(np.mean(flow_variances)),
            "occlusion_entropy": float(self._entropy(occlusion_map))
        }

        return uncertainty

    def _entropy(self, values: np.ndarray) -> float:
        hist, _ = np.histogram(values, bins=20, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist)))

    def _sanity_check(self, results: Dict[str, Any]) -> None:
        """
        Ensure outputs are numerically sane.
        """
        if np.isnan(results["motion_vectors"]).any():
            raise ValueError("NaN detected in motion vectors.")

        if results["uncertainty"]["mean_flow_variance"] < 0:
            raise ValueError("Negative variance detected.")

        if results["uncertainty"]["occlusion_entropy"] < 0:
            raise ValueError("Negative entropy detected.")
