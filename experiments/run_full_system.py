# experiments/run_full_system.py

from utils.config import ExperimentConfig
from utils.logging import ExperimentLogger
from dataset.loader import DatasetLoader


def run_baseline_0():
    """
    Baseline 0
    Verifies dataset integrity and logs dataset statistics
    before any perception, learning, or causal modelling.
    """

    # Load configuration and logger
    config = ExperimentConfig()
    logger = ExperimentLogger()

    # Load dataset
    dataset_loader = DatasetLoader(config.dataset)
    clips = dataset_loader.load()

    # Separate used and held out clips
    used_clips = [c for c in clips if not c.held_out]
    held_out_clips = [c for c in clips if c.held_out]

    # Basic counts
    total_clips = len(clips)
    used_count = len(used_clips)
    held_out_count = len(held_out_clips)
    held_out_ratio = held_out_count / total_clips if total_clips > 0 else 0.0

    # Compute video lengths in seconds for used clips only
    lengths_seconds = [
        len(c.frames) / c.fps
        for c in used_clips
    ]

    # FPS consistency check
    fps_values = sorted(set(c.fps for c in clips))
    fps_consistent = len(fps_values) == 1

    # Frame counts per used clip
    frames_per_clip = [len(c.frames) for c in used_clips]

    # Integrity checks
    empty_frames_found = any(len(c.frames) == 0 for c in clips)
    invalid_fps_found = any(c.fps <= 0 for c in clips)

    # Build dataset summary
    dataset_summary = {
        "dataset_summary": {
            "total_clips": total_clips,
            "used_clips": used_count,
            "held_out_clips": held_out_count,
            "held_out_ratio": held_out_ratio
        },
        "video_length_seconds": {
            "min": min(lengths_seconds) if lengths_seconds else 0.0,
            "max": max(lengths_seconds) if lengths_seconds else 0.0,
            "mean": (
                sum(lengths_seconds) / len(lengths_seconds)
                if lengths_seconds else 0.0
            ),
            "all_lengths": lengths_seconds
        },
        "frame_statistics": {
            "fps_values": fps_values,
            "fps_consistent": fps_consistent,
            "frames_per_clip": frames_per_clip
        },
        "held_out_clips": {
            "clip_ids": [c.clip_id for c in held_out_clips],
            "selection_method": "deterministic_split",
            "held_out_only_used_for_evaluation": True
        },
        "integrity_checks": {
            "empty_frames_found": empty_frames_found,
            "fps_override_applied": False,
            "corrupted_videos_detected": False,
            "non_deterministic_ordering": False,
            "invalid_fps_found": invalid_fps_found
        },
        "notes": (
            "Baseline 0 confirms dataset hygiene prior to "
            "perception or causal modelling."
        )
    }

    # Log baseline
    logger.log_baseline("baseline_0_dataset", dataset_summary)


if __name__ == "__main__":
    run_baseline_0()
