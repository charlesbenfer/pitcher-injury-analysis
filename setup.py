#!/usr/bin/env python3
"""Setup script for mlb_pitcher_videos package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="mlb_pitcher_videos",
    version="0.1.0",
    author="Charles Benfer",
    description="Tools for scraping and processing MLB pitcher footage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlesbenfer/mlb_pitcher_videos",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "requests>=2.25.0",
        "pybaseball>=2.2.0",
    ],
    extras_require={
        "pose": ["mediapipe>=0.10.0"],
        "notebooks": ["jupyter", "notebook", "ipywidgets"],
        "all": ["mediapipe>=0.10.0", "jupyter", "notebook", "ipywidgets", "tqdm"],
    },
    entry_points={
        "console_scripts": [
            "mlb-scrape=mlb_pitcher_videos.cli:scrape",
            "mlb-download=mlb_pitcher_videos.cli:download",
            "mlb-isolate=mlb_pitcher_videos.cli:isolate",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Multimedia :: Video",
    ],
    keywords="baseball mlb pitcher video scraping statcast",
)
