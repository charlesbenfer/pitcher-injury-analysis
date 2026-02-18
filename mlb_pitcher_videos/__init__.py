"""
MLB Pitcher Videos - Tools for scraping and processing MLB pitcher footage.

A toolkit for:
- Scraping Statcast pitch data from Baseball Savant
- Downloading pitch videos from MLB
- Isolating pitcher footage (cropping + trimming to pre-release)

Example:
    >>> from mlb_pitcher_videos import StatcastScraper, VideoDownloader, PitcherIsolator
    >>>
    >>> # Get pitch data for a pitcher
    >>> scraper = StatcastScraper()
    >>> pitches = scraper.get_pitcher_data(pitcher_id=608331, season=2024)
    >>>
    >>> # Download videos
    >>> downloader = VideoDownloader()
    >>> downloader.download_videos(pitches, output_dir="videos/")
    >>>
    >>> # Isolate pitcher in each video
    >>> isolator = PitcherIsolator()
    >>> isolator.process_directory("videos/", "videos_isolated/")
"""

__version__ = "0.1.0"
__author__ = "Charles Benfer"

from .scraper import StatcastScraper
from .downloader import VideoDownloader
from .isolator import PitcherIsolator

__all__ = [
    "StatcastScraper",
    "VideoDownloader",
    "PitcherIsolator",
]
