import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pybaseball as pyb
from pybaseball import statcast, playerid_lookup, pitching_stats
import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PitcherDataAcquisition:
    
    def __init__(self, start_year: int = 2015, end_year: int = 2024):
        self.start_year = start_year
        self.end_year = end_year
        self.injury_data = None
        self.pitcher_stats = None
        self.pitch_data = None
        
    def fetch_pitcher_list(self) -> pd.DataFrame:
        print(f"Fetching pitcher list from {self.start_year} to {self.end_year}...")
        
        all_pitchers = []
        for year in range(self.start_year, self.end_year + 1):
            print(f"  Fetching {year} season...")
            try:
                stats = pitching_stats(year, qual=50)
                stats['Season'] = year
                all_pitchers.append(stats)
                time.sleep(1)
            except Exception as e:
                print(f"    Error fetching {year}: {e}")
                continue
        
        if all_pitchers:
            pitchers_df = pd.concat(all_pitchers, ignore_index=True)
            print(f"Found {len(pitchers_df['Name'].unique())} unique pitchers")
            return pitchers_df
        else:
            return pd.DataFrame()
    
    def fetch_injury_data(self) -> pd.DataFrame:
        print("Fetching injury data from multiple sources...")
        
        injuries = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"  Processing {year} season injuries...")
            
            url = f"https://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate={year}-01-01&EndDate={year}-12-31&ILChkBx=yes&Submit=Search"
            
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                table = soup.find('table', {'class': 'datatable'})
                if table:
                    rows = table.find_all('tr')[1:]
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 5 and 'pitcher' in cols[2].text.lower():
                            injury_record = {
                                'Date': cols[0].text.strip(),
                                'Team': cols[1].text.strip(),
                                'Player': cols[2].text.strip(),
                                'Transaction': cols[3].text.strip(),
                                'Notes': cols[4].text.strip() if len(cols) > 4 else ''
                            }
                            injuries.append(injury_record)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"    Error fetching injuries for {year}: {e}")
                continue
        
        injuries_df = pd.DataFrame(injuries)
        
        if not injuries_df.empty:
            injuries_df['Date'] = pd.to_datetime(injuries_df['Date'], errors='coerce')
            
            injuries_df['InjuryType'] = injuries_df['Notes'].apply(self._classify_injury)
            injuries_df['DaysOnIL'] = self._calculate_il_duration(injuries_df)
            
            print(f"Found {len(injuries_df)} injury records")
        
        return injuries_df
    
    def _classify_injury(self, notes: str) -> str:
        notes_lower = notes.lower()
        
        injury_mapping = {
            'shoulder': ['shoulder', 'rotator cuff', 'labrum'],
            'elbow': ['elbow', 'ucl', 'tommy john', 'ulnar'],
            'forearm': ['forearm', 'flexor'],
            'back': ['back', 'spine', 'disc'],
            'oblique': ['oblique', 'side', 'rib'],
            'hamstring': ['hamstring'],
            'knee': ['knee'],
            'other': []
        }
        
        for injury_type, keywords in injury_mapping.items():
            if any(keyword in notes_lower for keyword in keywords):
                return injury_type
        
        return 'other'
    
    def _calculate_il_duration(self, df: pd.DataFrame) -> List[int]:
        durations = []
        
        for idx, row in df.iterrows():
            if 'placed' in row['Transaction'].lower():
                player_name = row['Player']
                placement_date = row['Date']
                
                return_mask = (
                    (df['Player'] == player_name) & 
                    (df['Date'] > placement_date) & 
                    (df['Transaction'].str.contains('activated|reinstated', case=False, na=False))
                )
                
                if return_mask.any():
                    return_date = df.loc[return_mask, 'Date'].min()
                    duration = (return_date - placement_date).days
                else:
                    duration = 60
                
                durations.append(duration)
            else:
                durations.append(0)
        
        return durations
    
    def fetch_pitch_metrics(self, pitcher_ids: List[int], sample_size: int = 100) -> pd.DataFrame:
        print(f"Fetching pitch-level metrics for {len(pitcher_ids)} pitchers...")
        
        pitch_data = []
        
        for i, pitcher_id in enumerate(pitcher_ids[:sample_size]):
            if i % 10 == 0:
                print(f"  Processing pitcher {i+1}/{min(len(pitcher_ids), sample_size)}...")
            
            try:
                for year in range(self.start_year, min(self.end_year + 1, 2024)):
                    start_date = f"{year}-04-01"
                    end_date = f"{year}-10-31"
                    
                    data = statcast(start_dt=start_date, end_dt=end_date, player_type='pitcher')
                    
                    if data is not None and not data.empty:
                        data = data[data['pitcher'] == pitcher_id]
                        
                        if not data.empty:
                            metrics = {
                                'pitcher_id': pitcher_id,
                                'year': year,
                                'avg_velocity': data['release_speed'].mean(),
                                'max_velocity': data['release_speed'].max(),
                                'velocity_std': data['release_speed'].std(),
                                'avg_spin_rate': data['release_spin_rate'].mean(),
                                'release_extension': data['release_extension'].mean(),
                                'pitch_count': len(data)
                            }
                            pitch_data.append(metrics)
                    
                    time.sleep(1)
                    
            except Exception as e:
                print(f"    Error fetching data for pitcher {pitcher_id}: {e}")
                continue
        
        pitch_df = pd.DataFrame(pitch_data)
        print(f"Collected pitch metrics for {len(pitch_df['pitcher_id'].unique())} pitchers")
        
        return pitch_df
    
    def create_survival_dataset(self) -> pd.DataFrame:
        print("Creating survival analysis dataset...")
        
        if self.pitcher_stats is None:
            self.pitcher_stats = self.fetch_pitcher_list()
        
        if self.injury_data is None:
            self.injury_data = self.fetch_injury_data()
        
        survival_data = []
        
        for _, pitcher in self.pitcher_stats.iterrows():
            pitcher_name = pitcher['Name']
            season = pitcher['Season']
            
            injuries = self.injury_data[
                (self.injury_data['Player'].str.contains(pitcher_name, case=False, na=False)) &
                (self.injury_data['Date'].dt.year == season)
            ]
            
            if not injuries.empty:
                first_injury = injuries.iloc[0]
                time_to_event = (first_injury['Date'] - pd.Timestamp(f"{season}-04-01")).days
                event_occurred = 1
                injury_type = first_injury['InjuryType']
                severity = first_injury['DaysOnIL']
            else:
                time_to_event = 180
                event_occurred = 0
                injury_type = None
                severity = 0
            
            record = {
                'player_name': pitcher_name,
                'season': season,
                'time_to_event': max(1, time_to_event),
                'event': event_occurred,
                'injury_type': injury_type,
                'severity': severity,
                'age': pitcher.get('Age', np.nan),
                'innings_pitched': pitcher.get('IP', np.nan),
                'era': pitcher.get('ERA', np.nan),
                'whip': pitcher.get('WHIP', np.nan),
                'k_per_9': pitcher.get('K/9', np.nan),
                'bb_per_9': pitcher.get('BB/9', np.nan),
                'hr_per_9': pitcher.get('HR/9', np.nan),
                'gb_percent': pitcher.get('GB%', np.nan)
            }
            
            survival_data.append(record)
        
        survival_df = pd.DataFrame(survival_data)
        
        survival_df = survival_df.dropna(subset=['time_to_event'])
        survival_df['time_to_event'] = survival_df['time_to_event'].clip(lower=1)
        
        print(f"Created survival dataset with {len(survival_df)} observations")
        print(f"Event rate: {survival_df['event'].mean():.2%}")
        
        return survival_df
    
    def save_data(self, output_dir: str = "data/processed"):
        print(f"Saving data to {output_dir}...")
        
        if self.pitcher_stats is not None:
            self.pitcher_stats.to_csv(f"{output_dir}/pitcher_stats.csv", index=False)
            print(f"  Saved pitcher_stats.csv")
        
        if self.injury_data is not None:
            self.injury_data.to_csv(f"{output_dir}/injury_data.csv", index=False)
            print(f"  Saved injury_data.csv")
        
        if self.pitch_data is not None:
            self.pitch_data.to_csv(f"{output_dir}/pitch_metrics.csv", index=False)
            print(f"  Saved pitch_metrics.csv")


if __name__ == "__main__":
    print("Starting MLB Pitcher Injury Data Acquisition")
    print("=" * 50)
    
    acquisition = PitcherDataAcquisition(start_year=2020, end_year=2023)
    
    pitchers = acquisition.fetch_pitcher_list()
    acquisition.pitcher_stats = pitchers
    
    injuries = acquisition.fetch_injury_data()
    acquisition.injury_data = injuries
    
    survival_df = acquisition.create_survival_dataset()
    
    survival_df.to_csv("data/processed/survival_dataset.csv", index=False)
    print("\nSurvival dataset saved to data/processed/survival_dataset.csv")
    
    acquisition.save_data()
    
    print("\nData acquisition complete!")
    print(f"Dataset shape: {survival_df.shape}")
    print(f"\nFirst 5 rows:")
    print(survival_df.head())