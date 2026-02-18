"""
Test the enhanced injury scraper with a small date range
"""

import sys
sys.path.append('.')

from scripts.enhanced_injury_scraper import MLBInjuryScraperEnhanced

# Test with just 2024 data
scraper = MLBInjuryScraperEnhanced()

# Test just IL data scraping for 2024
print("Testing IL data scraping for 2024...")
il_data, teams = scraper.scrape_il_data(2024, 2024)

print(f"IL data shape: {il_data.shape}")
print(f"Teams shape: {teams.shape}")
print("\nFirst few IL records:")
print(il_data.head())

print("\nTeam info:")
print(teams.head())