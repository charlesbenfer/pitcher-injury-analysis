#!/usr/bin/env python3
"""
Isolate pitcher from broadcast videos.

Crops and trims videos to focus on just the pitcher during pre-release phase.

Usage:
    python isolate_pitcher.py --input data/videos/ --output data/videos_isolated/
    python isolate_pitcher.py --input data/videos/ --output data/videos_isolated/ --max-duration 2.5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from mlb_pitcher_videos import PitcherIsolator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Isolate pitcher from broadcast baseball videos"
    )

    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Input video file or directory'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='data/videos_isolated/',
        help='Output directory (default: data/videos_isolated/)'
    )
    parser.add_argument(
        '--max-duration', type=float, default=2.5,
        help='Maximum duration in seconds (default: 2.5)'
    )
    parser.add_argument(
        '--max-videos', type=int,
        help='Maximum number of videos to process'
    )
    parser.add_argument(
        '--no-crop', action='store_true',
        help="Don't crop to pitcher region (keep full frame)"
    )
    parser.add_argument(
        '--no-filter', action='store_true',
        help="Don't filter to pitching view only"
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help='Checkpoint file for resumable processing'
    )
    parser.add_argument(
        '--use-pose', action='store_true',
        help='Use MediaPipe pose detection (slower but more precise)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Initialize isolator
    isolator = PitcherIsolator(use_pose_detection=args.use_pose)

    if input_path.is_file():
        # Process single video
        output_file = output_path / f"{input_path.stem}_isolated.mp4"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {input_path}")
        metadata = isolator.process_video(
            input_path,
            output_file,
            max_duration=args.max_duration,
            crop=not args.no_crop,
            filter_views=not args.no_filter,
        )

        print(f"\nOutput:")
        print(f"  File: {metadata['output']}")
        print(f"  Size: {metadata['output_size'][0]}x{metadata['output_size'][1]}")
        print(f"  Frames: {metadata['output_frames']}")
        print(f"  Duration: {metadata['duration']:.2f}s")

    else:
        # Process directory
        video_count = len(list(input_path.glob('*.mp4')))
        print(f"Found {video_count} videos in {input_path}")

        if args.max_videos:
            print(f"Processing up to {args.max_videos} videos")

        results = isolator.process_directory(
            input_path,
            output_path,
            max_duration=args.max_duration,
            max_videos=args.max_videos,
            checkpoint_file=args.checkpoint,
            crop=not args.no_crop,
            filter_views=not args.no_filter,
        )

        print(f"\nProcessed {len(results)} videos")
        print(f"Output directory: {output_path}")


if __name__ == '__main__':
    main()
