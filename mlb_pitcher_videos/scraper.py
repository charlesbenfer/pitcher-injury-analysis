"""
Statcast data scraper for MLB pitch data.

Uses pybaseball to fetch pitch-level data from Baseball Savant.
"""

import pandas as pd
from typing import Optional, List, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StatcastScraper:
    """
    Scrape Statcast pitch data from Baseball Savant.

    Example:
        >>> scraper = StatcastScraper()
        >>>
        >>> # Get all pitches for a specific pitcher in 2024
        >>> data = scraper.get_pitcher_data(pitcher_id=608331, season=2024)
        >>>
        >>> # Get pitches for specific pitch types
        >>> fastballs = scraper.get_pitcher_data(
        ...     pitcher_id=608331,
        ...     season=2024,
        ...     pitch_types=['FF', 'SI', 'FC']
        ... )
        >>>
        >>> # Get data for a date range
        >>> data = scraper.get_pitcher_data(
        ...     pitcher_id=608331,
        ...     start_date='2024-04-01',
        ...     end_date='2024-09-30'
        ... )
    """

    # Common pitch type codes
    PITCH_TYPES = {
        'FF': 'Four-Seam Fastball',
        'SI': 'Sinker',
        'FC': 'Cutter',
        'SL': 'Slider',
        'CU': 'Curveball',
        'CH': 'Changeup',
        'FS': 'Splitter',
        'KC': 'Knuckle Curve',
        'ST': 'Sweeper',
        'SV': 'Slurve',
        'KN': 'Knuckleball',
    }

    def __init__(self):
        """Initialize the scraper."""
        try:
            import pybaseball
            pybaseball.cache.enable()
            self._pybaseball = pybaseball
        except ImportError:
            raise ImportError(
                "pybaseball is required. Install with: pip install pybaseball"
            )

    def get_pitcher_data(
        self,
        pitcher_id: int,
        season: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pitch_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get Statcast data for a specific pitcher.

        Args:
            pitcher_id: MLB player ID (find at baseballsavant.mlb.com)
            season: Season year (e.g., 2024). If provided, overrides dates.
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            pitch_types: List of pitch type codes to filter (e.g., ['FF', 'SL'])

        Returns:
            DataFrame with pitch-level Statcast data
        """
        # Determine date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
        elif not start_date or not end_date:
            # Default to current season
            year = datetime.now().year
            start_date = f"{year}-03-01"
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching data for pitcher {pitcher_id} from {start_date} to {end_date}")

        # Fetch data
        data = self._pybaseball.statcast_pitcher(
            start_dt=start_date,
            end_dt=end_date,
            player_id=pitcher_id
        )

        if data.empty:
            logger.warning(f"No data found for pitcher {pitcher_id}")
            return data

        logger.info(f"Retrieved {len(data)} pitches")

        # Filter pitch types if specified
        if pitch_types:
            data = data[data['pitch_type'].isin(pitch_types)]
            logger.info(f"Filtered to {len(data)} pitches of types: {pitch_types}")

        # Sort by game date and pitch number
        data = data.sort_values(['game_date', 'at_bat_number', 'pitch_number'])

        return data

    def get_pitcher_info(self, pitcher_id: int) -> dict:
        """
        Get basic info about a pitcher.

        Args:
            pitcher_id: MLB player ID

        Returns:
            Dictionary with pitcher info
        """
        try:
            lookup = self._pybaseball.playerid_reverse_lookup([pitcher_id], key_type='mlbam')
            if not lookup.empty:
                row = lookup.iloc[0]
                return {
                    'id': pitcher_id,
                    'name': f"{row['name_first']} {row['name_last']}",
                    'first_name': row['name_first'],
                    'last_name': row['name_last'],
                }
        except Exception as e:
            logger.warning(f"Could not lookup pitcher info: {e}")

        return {'id': pitcher_id, 'name': 'Unknown'}

    def search_pitcher(self, name: str) -> pd.DataFrame:
        """
        Search for a pitcher by name.

        Args:
            name: Pitcher name (partial match supported)

        Returns:
            DataFrame with matching players
        """
        try:
            results = self._pybaseball.playerid_lookup(
                name.split()[-1],  # Last name
                name.split()[0] if len(name.split()) > 1 else None  # First name
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return pd.DataFrame()

    def prepare_for_download(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare Statcast data for video downloading.

        Adds play_id column and filters to rows with valid video identifiers.

        Args:
            data: Statcast DataFrame

        Returns:
            DataFrame ready for video download
        """
        # Create unique play identifier
        if 'sv_id' in data.columns:
            data = data.copy()
            # sv_id format: YYMMDD_HHMMSS
            # We'll create a unique play_id from game_pk and sv_id
            data['play_id'] = data.apply(
                lambda r: f"{r['game_pk']}_{r['sv_id']}" if pd.notna(r['sv_id']) else None,
                axis=1
            )
            data = data.dropna(subset=['play_id'])

        return data

    def save_data(self, data: pd.DataFrame, output_path: str):
        """
        Save Statcast data to CSV.

        Args:
            data: DataFrame to save
            output_path: Output file path
        """
        data.to_csv(output_path, index=False)
        logger.info(f"Saved {len(data)} rows to {output_path}")


def lookup_pitcher_id(name: str) -> Optional[int]:
    """
    Convenience function to look up a pitcher's MLB ID by name.

    Args:
        name: Pitcher name (e.g., "Max Fried" or "Fried")

    Returns:
        MLB player ID or None if not found
    """
    scraper = StatcastScraper()
    results = scraper.search_pitcher(name)

    if results.empty:
        print(f"No results found for '{name}'")
        return None

    print(f"\nFound {len(results)} results:")
    print(results[['name_first', 'name_last', 'key_mlbam']].to_string(index=False))

    if len(results) == 1:
        return int(results.iloc[0]['key_mlbam'])

    return None
