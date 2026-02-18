#!/usr/bin/env python3
"""
Full pipeline: Scrape -> Download -> Isolate

Run all steps in sequence for a given pitcher.

Usage:
    python full_pipeline.py --pitcher-id 608331 --season 2024
    python full_pipeline.py --pitcher-name "Max Fried" --season 2024 --max-videos 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from mlb_pitcher_videos import StatcastScraper, VideoDownloader, PitcherIsolator
from mlb_pitcher_videos.scraper import lookup_pitcher_id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Scrape -> Download -> Isolate"
    )

    # Pitcher identification
    pitcher_group = parser.add_mutually_exclusive_group(required=True)
    pitcher_group.add_argument('--pitcher-id', type=int, help='MLB player ID')
    pitcher_group.add_argument('--pitcher-name', type=str, help='Pitcher name')

    # Date range
    parser.add_argument('--season', type=int, help='Season year')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')

    # Filters
    parser.add_argument(
        '--pitch-types', type=str, nargs='+',
        help='Pitch types to include (e.g., FF SL CU)'
    )

    # Limits
    parser.add_argument('--max-videos', type=int, help='Maximum videos to process')

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='data/',
        help='Base output directory (default: data/)'
    )

    # Options
    parser.add_argument(
        '--max-duration', type=float, default=2.5,
        help='Max duration for isolated videos (default: 2.5s)'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip download step (use existing videos)'
    )
    parser.add_argument(
        '--skip-isolate', action='store_true',
        help='Skip isolation step'
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(args.output_dir)
    statcast_file = base_dir / 'statcast.csv'
    videos_dir = base_dir / 'videos'
    isolated_dir = base_dir / 'videos_isolated'

    # Get pitcher ID
    if args.pitcher_name:
        pitcher_id = lookup_pitcher_id(args.pitcher_name)
        if not pitcher_id:
            print(f"Could not find pitcher: {args.pitcher_name}")
            sys.exit(1)
    else:
        pitcher_id = args.pitcher_id

    # Validate dates
    if not args.season and not (args.start_date and args.end_date):
        parser.error("Either --season or both --start-date and --end-date required")

    # =========================================================================
    # STEP 1: Scrape Statcast data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Scraping Statcast data")
    print("=" * 60)

    scraper = StatcastScraper()
    info = scraper.get_pitcher_info(pitcher_id)
    print(f"Pitcher: {info.get('name', 'Unknown')} (ID: {pitcher_id})")

    data = scraper.get_pitcher_data(
        pitcher_id=pitcher_id,
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        pitch_types=args.pitch_types,
    )

    if data.empty:
        print("No data found!")
        sys.exit(1)

    statcast_file.parent.mkdir(parents=True, exist_ok=True)
    scraper.save_data(data, str(statcast_file))

    print(f"Found {len(data)} pitches")
    print(f"Pitch types: {data['pitch_type'].value_counts().to_dict()}")

    # =========================================================================
    # STEP 2: Download videos
    # =========================================================================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading videos")
        print("=" * 60)

        downloader = VideoDownloader()
        results = downloader.download_from_dataframe(
            data,
            output_dir=videos_dir,
            max_videos=args.max_videos,
        )

        print(f"Downloaded: {results['successful']}/{results['total']}")
    else:
        print("\n[Skipping download step]")

    # =========================================================================
    # STEP 3: Isolate pitcher
    # =========================================================================
    if not args.skip_isolate:
        print("\n" + "=" * 60)
        print("STEP 3: Isolating pitcher from videos")
        print("=" * 60)

        if not videos_dir.exists():
            print(f"Videos directory not found: {videos_dir}")
            sys.exit(1)

        isolator = PitcherIsolator()
        results = isolator.process_directory(
            videos_dir,
            isolated_dir,
            max_duration=args.max_duration,
            max_videos=args.max_videos,
        )

        print(f"Processed: {len(results)} videos")
    else:
        print("\n[Skipping isolation step]")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Statcast data: {statcast_file}")
    print(f"Raw videos: {videos_dir}")
    print(f"Isolated videos: {isolated_dir}")

    # Count files
    if videos_dir.exists():
        video_count = len(list(videos_dir.glob('*.mp4')))
        print(f"  Raw video count: {video_count}")

    if isolated_dir.exists():
        isolated_count = len(list(isolated_dir.glob('*.mp4')))
        print(f"  Isolated video count: {isolated_count}")


if __name__ == '__main__':
    main()
