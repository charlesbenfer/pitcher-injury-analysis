#!/usr/bin/env python3
"""
Download pitch videos from Baseball Savant.

Usage:
    python download_videos.py --input data/statcast.csv --output data/videos/
    python download_videos.py --input data/statcast.csv --output data/videos/ --max-videos 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from mlb_pitcher_videos import VideoDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Download pitch videos from Baseball Savant"
    )

    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Input CSV file with Statcast data'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='data/videos/',
        help='Output directory for videos (default: data/videos/)'
    )
    parser.add_argument(
        '--max-videos', type=int,
        help='Maximum number of videos to download'
    )
    parser.add_argument(
        '--delay', type=float, default=0.5,
        help='Delay between requests in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Use parallel downloads (faster but may hit rate limits)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    data = pd.read_csv(input_path)
    print(f"Found {len(data)} pitches")

    # Check required columns
    required_cols = ['game_pk', 'sv_id', 'pitch_type']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)

    # Download
    downloader = VideoDownloader(delay=args.delay)

    print(f"\nDownloading videos to: {args.output}")
    if args.max_videos:
        print(f"Limiting to {args.max_videos} videos")

    results = downloader.download_from_dataframe(
        data,
        output_dir=args.output,
        max_videos=args.max_videos,
        parallel=args.parallel,
        workers=args.workers,
    )

    # Summary
    print(f"\nDownload Summary:")
    print(f"  Total attempted: {results['total']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped (already existed): {results['skipped']}")
    print(f"  Output directory: {args.output}")


if __name__ == '__main__':
    main()
