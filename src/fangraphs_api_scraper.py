import pandas as pd
import requests
import time
from datetime import datetime
import json
from typing import Dict, List

class FanGraphsAPIScraper:
    
    def __init__(self):
        self.base_url = "https://www.fangraphs.com/api/roster-resource/injury-report/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.fangraphs.com/roster-resource/injury-report',
            'Accept': 'application/json, text/plain, */*'
        }
    
    def fetch_injury_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch injury data for specified seasons using the FanGraphs API
        """
        all_injuries = []
        
        for season in seasons:
            print(f"Fetching injury data for {season}...")
            
            # Use current timestamp as loaddate (seems to be a cache-busting parameter)
            loaddate = int(time.time())
            
            params = {
                'season': season,
                'loaddate': loaddate
            }
            
            try:
                response = requests.get(self.base_url, params=params, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        print(f"  Found {len(data)} injury records for {season}")
                        
                        for record in data:
                            # Add season to each record
                            record['season'] = season
                            all_injuries.append(record)
                    
                    elif isinstance(data, dict):
                        # Handle different response formats
                        if 'data' in data:
                            records = data['data']
                            print(f"  Found {len(records)} injury records for {season}")
                            for record in records:
                                record['season'] = season
                                all_injuries.append(record)
                        else:
                            print(f"  Unexpected data format for {season}: {list(data.keys())}")
                    
                else:
                    print(f"  Failed to fetch {season}: HTTP {response.status_code}")
                
            except Exception as e:
                print(f"  Error fetching {season}: {e}")
            
            # Be respectful with API requests
            time.sleep(2)
        
        if all_injuries:
            df = pd.DataFrame(all_injuries)
            return self._process_injury_dataframe(df)
        else:
            print("No injury data found via API")
            return pd.DataFrame()
    
    def _process_injury_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the injury dataframe
        """
        print(f"Processing {len(df)} injury records...")
        
        # Display column information
        print(f"Columns found: {list(df.columns)}")
        print(f"Sample record:")
        if len(df) > 0:
            print(df.iloc[0].to_dict())
        
        # Filter for pitchers only
        pitcher_positions = ['SP', 'RP', 'P', 'LHP', 'RHP', 'CL']  # Include closers
        
        # Try different possible column names for position
        position_cols = ['pos', 'position', 'Position', 'Pos']
        position_col = None
        
        for col in position_cols:
            if col in df.columns:
                position_col = col
                break
        
        if position_col:
            pitcher_df = df[df[position_col].isin(pitcher_positions)].copy()
            print(f"Found {len(pitcher_df)} pitcher injury records")
        else:
            print("No position column found - keeping all records")
            pitcher_df = df.copy()
        
        # Process dates if present
        date_columns = ['date', 'il_retro_date', 'eligible_to_return', 'return_date', 'latest_update']
        
        for col in date_columns:
            if col in pitcher_df.columns:
                # Convert dates, handling various formats
                pitcher_df[col] = pd.to_datetime(pitcher_df[col], errors='coerce')
        
        # Clean and categorize injury types
        if 'injury' in pitcher_df.columns or 'Injury/Surgery' in pitcher_df.columns:
            injury_col = 'injury' if 'injury' in pitcher_df.columns else 'Injury/Surgery'
            pitcher_df['injury_category'] = pitcher_df[injury_col].apply(self._categorize_injury)
        
        return pitcher_df
    
    def _categorize_injury(self, injury_text: str) -> str:
        """
        Categorize injury descriptions into standard categories
        """
        if pd.isna(injury_text):
            return 'unknown'
        
        injury_lower = str(injury_text).lower()
        
        if any(term in injury_lower for term in ['elbow', 'ucl', 'tommy john', 'ulnar']):
            return 'elbow'
        elif any(term in injury_lower for term in ['shoulder', 'rotator', 'labrum']):
            return 'shoulder'
        elif any(term in injury_lower for term in ['forearm', 'flexor']):
            return 'forearm'
        elif any(term in injury_lower for term in ['back', 'spine', 'disc']):
            return 'back'
        elif 'oblique' in injury_lower:
            return 'oblique'
        elif any(term in injury_lower for term in ['hamstring', 'quad', 'calf', 'leg']):
            return 'leg'
        elif any(term in injury_lower for term in ['hand', 'finger', 'wrist']):
            return 'hand'
        else:
            return 'other'
    
    def save_data(self, df: pd.DataFrame, filename: str = 'fangraphs_real_injuries.csv'):
        """
        Save injury data to CSV
        """
        output_path = f"data/processed/{filename}"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} injury records to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['season'].min()} - {df['season'].max()}")
        
        if 'injury_category' in df.columns:
            print(f"\nInjury categories:")
            print(df['injury_category'].value_counts())
        
        if 'pos' in df.columns:
            print(f"\nPositions:")
            print(df['pos'].value_counts())


if __name__ == "__main__":
    print("FanGraphs API Injury Data Scraper")
    print("=" * 50)
    
    scraper = FanGraphsAPIScraper()
    
    # Fetch data for recent seasons
    # Start with 2024-2025, expand if needed
    seasons = [2024, 2025]
    
    injury_df = scraper.fetch_injury_data(seasons)
    
    if not injury_df.empty:
        scraper.save_data(injury_df)
        
        print(f"\nSample of scraped data:")
        print(injury_df.head())
    else:
        print("No injury data could be scraped")