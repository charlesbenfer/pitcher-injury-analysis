# MLB Pitcher Videos

A Python toolkit for scraping, downloading, and processing MLB pitcher footage from Baseball Savant.

Perfect for:
- **Pitch tipping analysis** - Detect tells in pitcher mechanics
- **Biomechanics research** - Study pitcher movements and form
- **Machine learning datasets** - Build training data for pitch classification
- **Scouting applications** - Analyze pitcher tendencies

## Features

- **Statcast Scraping** - Fetch detailed pitch data from Baseball Savant
- **Video Downloading** - Download pitch clips for any pitcher/season
- **Pitcher Isolation** - Crop & trim videos to focus on the pitcher pre-release
- **Jupyter Notebooks** - Interactive tutorials for non-coders

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlb_pitcher_videos.git
cd mlb_pitcher_videos

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Basic Usage

```python
from mlb_pitcher_videos import StatcastScraper, VideoDownloader, PitcherIsolator

# 1. Get pitch data for a pitcher
scraper = StatcastScraper()
pitches = scraper.get_pitcher_data(
    pitcher_id=608331,  # Max Fried
    season=2024,
    pitch_types=['FF', 'CU', 'SL']  # Optional: filter pitch types
)
print(f"Found {len(pitches)} pitches")

# 2. Download the videos
downloader = VideoDownloader()
results = downloader.download_from_dataframe(
    pitches,
    output_dir="data/videos/",
    max_videos=100  # Optional: limit downloads
)
print(f"Downloaded {results['successful']} videos")

# 3. Isolate pitcher (crop + trim to pre-release)
isolator = PitcherIsolator()
isolator.process_directory(
    "data/videos/",
    "data/videos_isolated/",
    max_duration=2.5  # Keep only first 2.5 seconds
)
```

### Using Scripts

```bash
# Scrape data for a pitcher
python scripts/scrape_data.py --pitcher-id 608331 --season 2024

# Download videos
python scripts/download_videos.py --input data/statcast.csv --output data/videos/

# Isolate pitcher in videos
python scripts/isolate_pitcher.py --input data/videos/ --output data/videos_isolated/

# Run the full pipeline
python scripts/full_pipeline.py --pitcher-id 608331 --season 2024
```

### Using Notebooks (No Coding Required!)

Open the notebooks in Jupyter for a guided, interactive experience:

```bash
jupyter notebook notebooks/
```

1. **`01_getting_started.ipynb`** - Overview and setup
2. **`02_scrape_statcast_data.ipynb`** - Get pitch data for any pitcher
3. **`03_download_videos.ipynb`** - Download video clips
4. **`04_isolate_pitcher.ipynb`** - Process videos to focus on pitcher

## Finding Pitcher IDs

You can look up any pitcher's MLB ID:

```python
from mlb_pitcher_videos.scraper import lookup_pitcher_id

# Search by name
lookup_pitcher_id("Max Fried")
# Output: Found 1 results:
#   name_first name_last  key_mlbam
#         Max     Fried     608331
```

Or find IDs on [Baseball Savant](https://baseballsavant.mlb.com/):
1. Search for a player
2. The ID is in the URL: `baseballsavant.mlb.com/savant-player/max-fried-608331`

## Common Pitch Type Codes

| Code | Pitch Type |
|------|------------|
| FF | Four-Seam Fastball |
| SI | Sinker |
| FC | Cutter |
| SL | Slider |
| CU | Curveball |
| CH | Changeup |
| FS | Splitter |
| ST | Sweeper |
| KC | Knuckle Curve |

## Output Structure

After running the full pipeline:

```
data/
├── statcast.csv          # Pitch-level data
├── videos/               # Raw downloaded videos
│   ├── abc123_FF.mp4
│   ├── def456_CU.mp4
│   └── ...
└── videos_isolated/      # Processed videos (cropped, trimmed)
    ├── abc123_FF_isolated.mp4
    ├── def456_CU_isolated.mp4
    └── ...
```

## Video Processing Details

The **pitcher isolation** step:

1. **Detects pitching camera** - Filters out replays, crowd shots, other angles
2. **Crops to pitcher** - Focuses on the mound area (removes batter, crowd)
3. **Trims to pre-release** - Keeps only the windup/set through release

This gives you clean, consistent footage of just the pitcher's mechanics.

### Crop Region

Default crop (center-field camera view):
- Horizontal: 25% to 75% of frame width
- Vertical: 35% to 95% of frame height

Customize if needed:

```python
isolator = PitcherIsolator(crop_region={
    'x_start': 0.20,
    'x_end': 0.80,
    'y_start': 0.30,
    'y_end': 0.95,
})
```

## API Reference

### StatcastScraper

```python
scraper = StatcastScraper()

# Get pitcher data
data = scraper.get_pitcher_data(
    pitcher_id=608331,
    season=2024,              # Or use start_date/end_date
    pitch_types=['FF', 'SL'], # Optional filter
)

# Search for a pitcher
results = scraper.search_pitcher("Fried")

# Get pitcher info
info = scraper.get_pitcher_info(608331)
```

### VideoDownloader

```python
downloader = VideoDownloader(
    delay=0.5,       # Seconds between requests
    max_retries=3,   # Retry failed downloads
)

# Download from Statcast dataframe
results = downloader.download_from_dataframe(
    data,
    output_dir="videos/",
    max_videos=100,
    parallel=False,  # Set True for faster (but be nice to servers)
)

# Download single video
downloader.download_video(
    game_pk=717465,
    play_id="230401_123456",
    pitch_type="FF",
    output_dir="videos/",
)
```

### PitcherIsolator

```python
isolator = PitcherIsolator(
    crop_region=None,          # Custom crop (or use default)
    use_pose_detection=False,  # Use MediaPipe (slower but more precise)
)

# Process single video
metadata = isolator.process_video(
    "input.mp4",
    "output.mp4",
    max_duration=2.5,
    crop=True,
    filter_views=True,
)

# Process directory
results = isolator.process_directory(
    "videos/",
    "videos_isolated/",
    max_duration=2.5,
    checkpoint_file="checkpoint.json",  # Enable resume
)
```

## Tips

### Rate Limiting
Be respectful of Baseball Savant's servers:
- Use the default delay (0.5s between requests)
- Don't download thousands of videos at once
- Consider running overnight for large batches

### Storage
Videos are ~2-5 MB each. Plan accordingly:
- 100 videos ≈ 300 MB
- 500 videos ≈ 1.5 GB
- 1000 videos ≈ 3 GB

### Video Quality
Downloaded videos are 1280x720 at ~60 FPS. After isolation:
- Resolution: 640x432 (cropped to pitcher)
- Duration: ~2.5 seconds (pre-release only)
- ~60-150 frames per video

## Requirements

- Python 3.8+
- OpenCV
- pandas
- pybaseball
- requests
- mediapipe (optional, for advanced pose detection)

## License

MIT License - feel free to use for research, personal projects, or commercial applications.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- [pybaseball](https://github.com/jldbc/pybaseball) for Statcast data access
- [Baseball Savant](https://baseballsavant.mlb.com/) for the data and videos
- [MediaPipe](https://mediapipe.dev/) for pose detection
