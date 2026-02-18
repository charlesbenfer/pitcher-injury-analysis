"""
Video downloader for MLB pitch videos from Baseball Savant.

Downloads pitch video clips using play IDs from Statcast data.
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Download pitch videos from Baseball Savant.

    Example:
        >>> downloader = VideoDownloader()
        >>>
        >>> # Download a single video by play ID
        >>> downloader.download_video(
        ...     game_pk=717465,
        ...     play_id="230401_123456",
        ...     pitch_type="FF",
        ...     output_dir="videos/"
        ... )
        >>>
        >>> # Download all videos from Statcast data
        >>> downloader.download_from_dataframe(statcast_df, output_dir="videos/")
    """

    BASE_URL = "https://baseballsavant.mlb.com"
    SPORTY_URL = "https://sporty-clips.mlb.com"

    def __init__(self, delay: float = 0.5, max_retries: int = 3):
        """
        Initialize the downloader.

        Args:
            delay: Delay between requests in seconds (be nice to servers)
            max_retries: Maximum retry attempts for failed downloads
        """
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_video_url(self, game_pk: int, play_id: str) -> Optional[str]:
        """
        Get the video URL for a specific play.

        Args:
            game_pk: Game primary key
            play_id: Play identifier (sv_id from Statcast)

        Returns:
            Video URL or None if not found
        """
        # Try the sporty clips URL format first (most reliable)
        page_url = f"{self.BASE_URL}/sporty-videos?playId={play_id}"

        try:
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()

            # Look for video URL in page content
            # Pattern: https://sporty-clips.mlb.com/.../.../...mp4
            patterns = [
                r'(https://sporty-clips\.mlb\.com/[^"\'>\s]+\.mp4)',
                r'(https://[^"\'>\s]*mlb[^"\'>\s]*\.mp4)',
                r'video["\']?\s*:\s*["\']([^"\']+\.mp4)',
            ]

            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    return match.group(1)

        except requests.RequestException as e:
            logger.debug(f"Failed to get video URL for {play_id}: {e}")

        return None

    def download_video(
        self,
        game_pk: int,
        play_id: str,
        pitch_type: str,
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download a single pitch video.

        Args:
            game_pk: Game primary key
            play_id: Play identifier
            pitch_type: Pitch type code (e.g., 'FF', 'SL')
            output_dir: Directory to save video
            filename: Custom filename (default: {play_id}_{pitch_type}.mp4)

        Returns:
            Path to downloaded video or None if failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clean play_id for filename
        clean_play_id = play_id.split('_')[-1] if '_' in str(play_id) else str(play_id)[:8]
        filename = filename or f"{clean_play_id}_{pitch_type}.mp4"
        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"Already exists: {filename}")
            return output_path

        # Get video URL
        video_url = self.get_video_url(game_pk, play_id)
        if not video_url:
            logger.warning(f"Could not find video URL for {play_id}")
            return None

        # Download video
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(video_url, timeout=60, stream=True)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded: {filename}")
                return output_path

            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))

        return None

    def download_from_dataframe(
        self,
        data: pd.DataFrame,
        output_dir: Union[str, Path],
        max_videos: Optional[int] = None,
        parallel: bool = False,
        workers: int = 4,
    ) -> Dict[str, any]:
        """
        Download videos for all pitches in a Statcast DataFrame.

        Args:
            data: Statcast DataFrame with game_pk, sv_id, pitch_type columns
            output_dir: Directory to save videos
            max_videos: Maximum number of videos to download
            parallel: Use parallel downloads (faster but be careful with rate limits)
            workers: Number of parallel workers

        Returns:
            Dictionary with download statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare download list
        downloads = []
        for _, row in data.iterrows():
            if pd.isna(row.get('sv_id')) or pd.isna(row.get('game_pk')):
                continue

            downloads.append({
                'game_pk': int(row['game_pk']),
                'play_id': str(row['sv_id']),
                'pitch_type': str(row.get('pitch_type', 'UNK')),
            })

        if max_videos:
            downloads = downloads[:max_videos]

        logger.info(f"Downloading {len(downloads)} videos to {output_dir}")

        # Track results
        results = {
            'total': len(downloads),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'paths': [],
        }

        if parallel:
            results = self._download_parallel(downloads, output_dir, workers, results)
        else:
            results = self._download_sequential(downloads, output_dir, results)

        logger.info(
            f"Download complete: {results['successful']} successful, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )

        return results

    def _download_sequential(
        self,
        downloads: List[Dict],
        output_dir: Path,
        results: Dict,
    ) -> Dict:
        """Download videos sequentially."""
        for i, item in enumerate(downloads):
            logger.info(f"[{i+1}/{len(downloads)}] Downloading {item['play_id']}")

            path = self.download_video(
                game_pk=item['game_pk'],
                play_id=item['play_id'],
                pitch_type=item['pitch_type'],
                output_dir=output_dir,
            )

            if path:
                if path.stat().st_size > 1000:  # Actual download
                    results['successful'] += 1
                else:
                    results['skipped'] += 1
                results['paths'].append(str(path))
            else:
                results['failed'] += 1

            time.sleep(self.delay)

        return results

    def _download_parallel(
        self,
        downloads: List[Dict],
        output_dir: Path,
        workers: int,
        results: Dict,
    ) -> Dict:
        """Download videos in parallel."""
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.download_video,
                    item['game_pk'],
                    item['play_id'],
                    item['pitch_type'],
                    output_dir,
                ): item
                for item in downloads
            }

            for future in as_completed(futures):
                item = futures[future]
                try:
                    path = future.result()
                    if path:
                        results['successful'] += 1
                        results['paths'].append(str(path))
                    else:
                        results['failed'] += 1
                except Exception as e:
                    logger.error(f"Error downloading {item['play_id']}: {e}")
                    results['failed'] += 1

        return results


def download_pitcher_videos(
    pitcher_id: int,
    output_dir: str = "videos",
    season: Optional[int] = None,
    pitch_types: Optional[List[str]] = None,
    max_videos: Optional[int] = None,
) -> Dict:
    """
    Convenience function to download all videos for a pitcher.

    Args:
        pitcher_id: MLB player ID
        output_dir: Directory to save videos
        season: Season year
        pitch_types: List of pitch types to filter
        max_videos: Maximum videos to download

    Returns:
        Download statistics dictionary
    """
    from .scraper import StatcastScraper

    # Get pitch data
    scraper = StatcastScraper()
    data = scraper.get_pitcher_data(
        pitcher_id=pitcher_id,
        season=season,
        pitch_types=pitch_types,
    )

    if data.empty:
        logger.error("No pitch data found")
        return {'total': 0, 'successful': 0, 'failed': 0}

    # Download videos
    downloader = VideoDownloader()
    return downloader.download_from_dataframe(
        data,
        output_dir=output_dir,
        max_videos=max_videos,
    )
