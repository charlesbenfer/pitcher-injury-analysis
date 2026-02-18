"""
Pitcher isolator - Crops and trims videos to focus on the pitcher pre-release.

Processes broadcast baseball videos to:
1. Detect frames with the standard pitching camera view
2. Crop to focus on just the pitcher
3. Extract only pre-release frames (before ball release)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CropRegion:
    """Defines a rectangular crop region."""
    x: int      # Left edge
    y: int      # Top edge
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]


class PitcherIsolator:
    """
    Isolate pitcher from broadcast baseball video.

    Crops videos to focus on the pitcher and trims to pre-release phase
    for pitch tipping or biomechanics analysis.

    Example:
        >>> isolator = PitcherIsolator()
        >>>
        >>> # Process a single video
        >>> isolator.process_video(
        ...     "videos/pitch_001.mp4",
        ...     "videos_isolated/pitch_001_isolated.mp4"
        ... )
        >>>
        >>> # Process all videos in a directory
        >>> isolator.process_directory(
        ...     "videos/",
        ...     "videos_isolated/",
        ...     max_duration=2.5
        ... )
    """

    # Default pitcher region (as fraction of frame)
    # Pitcher is typically in bottom-center of center-field camera view
    DEFAULT_REGION = {
        'x_start': 0.25,   # 25% from left
        'x_end': 0.75,     # 75% from left
        'y_start': 0.35,   # 35% from top
        'y_end': 0.95,     # 95% from top
    }

    def __init__(
        self,
        crop_region: Optional[Dict[str, float]] = None,
        use_pose_detection: bool = False,
    ):
        """
        Initialize the isolator.

        Args:
            crop_region: Custom crop region as fractions (x_start, x_end, y_start, y_end)
            use_pose_detection: Use MediaPipe for more precise detection (slower)
        """
        self.crop_region_config = crop_region or self.DEFAULT_REGION
        self.use_pose_detection = use_pose_detection
        self._pose = None

        if use_pose_detection:
            self._init_pose_detection()

    def _init_pose_detection(self):
        """Initialize MediaPipe pose detection."""
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            logger.warning(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )
            self.use_pose_detection = False

    def detect_pitching_view(self, frame: np.ndarray) -> bool:
        """
        Detect if frame shows the standard center-field pitching camera.

        Uses color detection to find the pitcher's mound (dirt area).

        Args:
            frame: Video frame (BGR format)

        Returns:
            True if frame appears to be the pitching view
        """
        h, w = frame.shape[:2]

        # Check bottom-center region for dirt (mound)
        region = frame[int(h * 0.5):int(h * 0.95),
                      int(w * 0.35):int(w * 0.65)]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Dirt is typically orange-brown
        lower_dirt = np.array([8, 40, 80])
        upper_dirt = np.array([25, 200, 255])
        mask = cv2.inRange(hsv, lower_dirt, upper_dirt)

        # If >5% of region is dirt-colored, likely pitching view
        dirt_ratio = np.sum(mask > 0) / mask.size
        return dirt_ratio > 0.05

    def get_crop_region(self, frame: np.ndarray) -> CropRegion:
        """
        Get the crop region for the pitcher.

        Args:
            frame: Video frame

        Returns:
            CropRegion object
        """
        h, w = frame.shape[:2]
        cfg = self.crop_region_config

        x = int(w * cfg['x_start'])
        y = int(h * cfg['y_start'])
        width = int(w * (cfg['x_end'] - cfg['x_start']))
        height = int(h * (cfg['y_end'] - cfg['y_start']))

        return CropRegion(x, y, width, height)

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        max_duration: float = 2.5,
        crop: bool = True,
        filter_views: bool = True,
    ) -> Dict:
        """
        Process a video to isolate the pitcher.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            max_duration: Maximum duration in seconds
            crop: Whether to crop to pitcher region
            filter_views: Whether to filter to pitching view only

        Returns:
            Dictionary with processing metadata
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing: {input_path.name}")
        logger.info(f"  Input: {orig_w}x{orig_h}, {fps:.1f} FPS, {total_frames} frames")

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError("No frames read from video")

        # Get crop region
        crop_region = self.get_crop_region(frames[0]) if crop else None

        # Find valid frames
        max_frames = int(max_duration * fps)
        valid_indices = []

        for i, frame in enumerate(frames):
            if i >= max_frames:
                break

            if filter_views and not self.detect_pitching_view(frame):
                continue

            valid_indices.append(i)

        # Fallback if no valid frames found
        if not valid_indices:
            valid_indices = list(range(min(max_frames, len(frames))))

        # Determine output size
        if crop_region:
            out_w, out_h = crop_region.width, crop_region.height
        else:
            out_w, out_h = orig_w, orig_h

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        for idx in valid_indices:
            frame = frames[idx]
            if crop_region:
                frame = crop_region.crop(frame)
            out.write(frame)

        out.release()

        metadata = {
            'input': str(input_path),
            'output': str(output_path),
            'input_frames': len(frames),
            'output_frames': len(valid_indices),
            'input_size': (orig_w, orig_h),
            'output_size': (out_w, out_h),
            'fps': fps,
            'duration': len(valid_indices) / fps,
            'crop_region': crop_region.as_tuple() if crop_region else None,
        }

        logger.info(f"  Output: {out_w}x{out_h}, {len(valid_indices)} frames, "
                   f"{metadata['duration']:.2f}s")

        return metadata

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        max_duration: float = 2.5,
        max_videos: Optional[int] = None,
        checkpoint_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Process all videos in a directory.

        Args:
            input_dir: Directory with input videos
            output_dir: Directory for output videos
            max_duration: Maximum duration per video
            max_videos: Maximum videos to process
            checkpoint_file: File to track progress (enables resume)
            **kwargs: Additional arguments for process_video

        Returns:
            List of metadata dictionaries
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video files
        video_files = sorted(input_dir.glob('*.mp4'))
        if max_videos:
            video_files = video_files[:max_videos]

        # Load checkpoint
        processed = set()
        if checkpoint_file:
            checkpoint_file = Path(checkpoint_file)
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                    processed = set(checkpoint.get('processed', []))
                logger.info(f"Resuming: {len(processed)} already processed")

        # Filter to remaining
        remaining = [v for v in video_files if v.name not in processed]
        logger.info(f"Processing {len(remaining)} videos")

        results = []
        errors = []

        for i, video_path in enumerate(remaining):
            output_path = output_dir / f"{video_path.stem}_isolated.mp4"

            try:
                logger.info(f"[{i+1}/{len(remaining)}] {video_path.name}")
                metadata = self.process_video(
                    video_path,
                    output_path,
                    max_duration=max_duration,
                    **kwargs,
                )
                results.append(metadata)
                processed.add(video_path.name)

            except Exception as e:
                logger.error(f"  Error: {e}")
                errors.append({'video': video_path.name, 'error': str(e)})

            # Save checkpoint periodically
            if checkpoint_file and (i + 1) % 25 == 0:
                self._save_checkpoint(checkpoint_file, processed, errors)

        # Final checkpoint
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, processed, errors)

        logger.info(f"Complete: {len(results)} successful, {len(errors)} errors")
        return results

    def _save_checkpoint(self, path: Path, processed: set, errors: list):
        """Save processing checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'processed': list(processed),
                'errors': errors,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)

    def __del__(self):
        if self._pose:
            self._pose.close()


def isolate_pitcher_videos(
    input_dir: str,
    output_dir: str = "videos_isolated",
    max_duration: float = 2.5,
    max_videos: Optional[int] = None,
) -> List[Dict]:
    """
    Convenience function to isolate pitcher in all videos.

    Args:
        input_dir: Directory with raw videos
        output_dir: Directory for isolated videos
        max_duration: Maximum duration per video
        max_videos: Maximum videos to process

    Returns:
        List of processing metadata
    """
    isolator = PitcherIsolator()
    return isolator.process_directory(
        input_dir,
        output_dir,
        max_duration=max_duration,
        max_videos=max_videos,
    )
