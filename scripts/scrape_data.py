#!/usr/bin/env python3
"""
Scrape Statcast pitch data for a pitcher.

Usage:
    python scrape_data.py --pitcher-id 608331 --season 2024
    python scrape_data.py --pitcher-name "Max Fried" --season 2024
    python scrape_data.py --pitcher-id 608331 --start-date 2024-04-01 --end-date 2024-09-30
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from mlb_pitcher_videos import StatcastScraper
from mlb_pitcher_videos.scraper import lookup_pitcher_id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Statcast pitch data from Baseball Savant"
    )

    # Pitcher identification
    pitcher_group = parser.add_mutually_exclusive_group(required=True)
    pitcher_group.add_argument(
        '--pitcher-id', type=int,
        help='MLB player ID (e.g., 608331 for Max Fried)'
    )
    pitcher_group.add_argument(
        '--pitcher-name', type=str,
        help='Pitcher name to search (e.g., "Max Fried")'
    )

    # Date range
    parser.add_argument(
        '--season', type=int,
        help='Season year (e.g., 2024)'
    )
    parser.add_argument(
        '--start-date', type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str,
        help='End date (YYYY-MM-DD)'
    )

    # Filters
    parser.add_argument(
        '--pitch-types', type=str, nargs='+',
        help='Pitch types to include (e.g., FF SL CU)'
    )

    # Output
    parser.add_argument(
        '-o', '--output', type=str, default='data/statcast.csv',
        help='Output CSV file (default: data/statcast.csv)'
    )

    args = parser.parse_args()

    # Get pitcher ID
    if args.pitcher_name:
        pitcher_id = lookup_pitcher_id(args.pitcher_name)
        if not pitcher_id:
            print(f"Could not find pitcher: {args.pitcher_name}")
            sys.exit(1)
    else:
        pitcher_id = args.pitcher_id

    # Validate date arguments
    if not args.season and not (args.start_date and args.end_date):
        parser.error("Either --season or both --start-date and --end-date are required")

    # Scrape data
    scraper = StatcastScraper()

    print(f"\nScraping data for pitcher ID: {pitcher_id}")
    info = scraper.get_pitcher_info(pitcher_id)
    print(f"Pitcher: {info.get('name', 'Unknown')}")

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

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scraper.save_data(data, str(output_path))

    # Summary
    print(f"\nSummary:")
    print(f"  Total pitches: {len(data)}")
    print(f"  Pitch types: {data['pitch_type'].value_counts().to_dict()}")
    print(f"  Date range: {data['game_date'].min()} to {data['game_date'].max()}")
    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    main()
