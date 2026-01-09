"""
dataset/loader.py

Responsible for:
- Loading raw video clips
- Enforcing strict held-out splits
- Providing frame-accurate access
- Guaranteeing reproducibility
- Serving as the ONLY entry point to the dataset

No annotations are loaded.
No labels are exposed to the system.
"""

from __future__ import annotations

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Generator, Optional


class VideoClip:
    """
    Container for a single video clip with deterministic frame access.
    """

    def __init__(
        self,
        clip_id: str,
        video_path: str,
        metadata: Dict,
        held_out: bool
    ):
        self.clip_id: str = clip_id
        self.video_path: str = video_path
        self.metadata: Dict = metadata
        self.held_out: bool = held_out

        self._frames: Optional[List[np.ndarray]] = None
        self._fps: Optional[float] = None

    def load_frames(self) -> None:
        """
        Loads all frames into memory with deterministic ordering.
        """
        if self._frames is not None:
            return

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        frames: List[np.ndarray] = []
        fps = capture.get(cv2.CAP_PROP_FPS)

        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)

        capture.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames found in video: {self.video_path}")

        self._frames = frames
        self._fps = fps

    @property
    def frames(self) -> List[np.ndarray]:
        self.load_frames()
        assert self._frames is not None
        return self._frames

    @property
    def fps(self) -> float:
        self.load_frames()
        assert self._fps is not None
        return self._fps

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_frame(self, frame_index: int) -> np.ndarray:
        """
        Returns a specific frame by index.
        """
        if frame_index < 0 or frame_index >= self.num_frames:
            raise IndexError("Frame index out of bounds")
        return self.frames[frame_index]

    def get_timestamp(self, frame_index: int) -> float:
        """
        Returns timestamp in seconds for a given frame index.
        """
        return frame_index / self.fps


class DatasetLoader:
    """
    Main dataset loader enforcing strict separation and reproducibility.
    """

    def __init__(
        self,
        dataset_root: str,
        held_out_ratio: float = 0.2,
        random_seed: int = 42
    ):
        self.dataset_root: str = dataset_root
        self.videos_dir: str = os.path.join(dataset_root, "videos")
        self.metadata_dir: str = os.path.join(dataset_root, "metadata")

        self.held_out_ratio: float = held_out_ratio
        self.random_seed: int = random_seed

        self._clips: Dict[str, VideoClip] = {}

        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        Loads all clips and enforces a deterministic held-out split.
        """
        if not os.path.isdir(self.videos_dir):
            raise FileNotFoundError(f"Missing videos directory: {self.videos_dir}")

        if not os.path.isdir(self.metadata_dir):
            raise FileNotFoundError(f"Missing metadata directory: {self.metadata_dir}")

        video_files = sorted(
            [f for f in os.listdir(self.videos_dir) if f.endswith(".mp4")]
        )

        if len(video_files) == 0:
            raise RuntimeError("No video files found")

        rng = np.random.default_rng(self.random_seed)
        shuffled_indices = rng.permutation(len(video_files))

        num_held_out = int(len(video_files) * self.held_out_ratio)
        held_out_indices = set(shuffled_indices[:num_held_out])

        for index, video_file in enumerate(video_files):
            clip_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(self.videos_dir, video_file)

            metadata_path = os.path.join(self.metadata_dir, f"{clip_id}.json")
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            held_out = index in held_out_indices

            clip = VideoClip(
                clip_id=clip_id,
                video_path=video_path,
                metadata=metadata,
                held_out=held_out
            )

            self._clips[clip_id] = clip

    def get_all_clips(self) -> List[VideoClip]:
        """
        Returns all clips regardless of split.
        """
        return list(self._clips.values())

    def get_training_clips(self) -> List[VideoClip]:
        """
        Returns clips not marked as held-out.
        """
        return [clip for clip in self._clips.values() if not clip.held_out]

    def get_held_out_clips(self) -> List[VideoClip]:
        """
        Returns held-out clips only.
        """
        return [clip for clip in self._clips.values() if clip.held_out]

    def get_clip_by_id(self, clip_id: str) -> VideoClip:
        """
        Retrieves a specific clip by ID.
        """
        if clip_id not in self._clips:
            raise KeyError(f"Clip ID not found: {clip_id}")
        return self._clips[clip_id]

    def iterate_frames(
        self,
        clip: VideoClip
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Iterates over frames yielding:
        (frame_index, timestamp_seconds, frame_array)
        """
        for frame_index in range(clip.num_frames):
            timestamp = clip.get_timestamp(frame_index)
            frame = clip.get_frame(frame_index)
            yield frame_index, timestamp, frame

    def dataset_summary(self) -> Dict:
        """
        Returns a summary useful for Baseline 0 logging.
        """
        total_clips = len(self._clips)
        held_out_clips = len(self.get_held_out_clips())
        training_clips = len(self.get_training_clips())

        frame_counts = []
        fps_values = []

        for clip in self._clips.values():
            clip.load_frames()
            frame_counts.append(clip.num_frames)
            fps_values.append(clip.fps)

        return {
            "total_clips": total_clips,
            "training_clips": training_clips,
            "held_out_clips": held_out_clips,
            "held_out_ratio": self.held_out_ratio,
            "frame_count_mean": float(np.mean(frame_counts)),
            "frame_count_std": float(np.std(frame_counts)),
            "fps_mean": float(np.mean(fps_values)),
            "fps_std": float(np.std(fps_values))
        }
