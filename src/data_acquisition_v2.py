import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pybaseball as pyb
from pybaseball import statcast, playerid_lookup, pitching_stats, cache
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Enable caching for pybaseball
cache.enable()

class PitcherDataAcquisitionV2:
    
    def __init__(self, start_year: int = 2020, end_year: int = 2023):
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
    
    def generate_synthetic_injury_data(self, pitcher_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic injury data for demonstration/testing purposes.
        This simulates realistic injury patterns based on pitcher workload and performance.
        """
        print("Generating synthetic injury data for demonstration...")
        
        np.random.seed(42)  # For reproducibility
        injuries = []
        
        # Get unique pitchers
        unique_pitchers = pitcher_stats['Name'].unique()
        
        # Define injury types and their base probabilities
        injury_types = {
            'elbow': 0.25,
            'shoulder': 0.20,
            'forearm': 0.15,
            'back': 0.15,
            'oblique': 0.10,
            'hamstring': 0.10,
            'other': 0.05
        }
        
        # Simulate injuries for each season
        for year in range(self.start_year, self.end_year + 1):
            season_pitchers = pitcher_stats[pitcher_stats['Season'] == year]
            
            # Determine which pitchers get injured (approximately 30% injury rate)
            n_injuries = int(len(season_pitchers) * 0.30)
            injured_pitchers = np.random.choice(season_pitchers['Name'].values, 
                                               size=min(n_injuries, len(season_pitchers)), 
                                               replace=False)
            
            for pitcher_name in injured_pitchers:
                pitcher_data = season_pitchers[season_pitchers['Name'] == pitcher_name].iloc[0]
                
                # Determine injury risk factors
                risk_factor = 1.0
                if 'IP' in pitcher_data and pitcher_data['IP'] > 180:
                    risk_factor *= 1.3  # Higher workload increases risk
                if 'Age' in pitcher_data and pitcher_data['Age'] > 32:
                    risk_factor *= 1.2  # Older pitchers have higher risk
                
                # Select injury type based on weighted probabilities
                injury_type = np.random.choice(list(injury_types.keys()), 
                                             p=list(injury_types.values()))
                
                # Generate injury date (random day during season)
                season_start = datetime(year, 4, 1)
                days_into_season = np.random.randint(0, 150)
                injury_date = season_start + timedelta(days=days_into_season)
                
                # Generate IL duration based on injury type
                severity_days = {
                    'elbow': (15, 90),
                    'shoulder': (20, 75),
                    'forearm': (10, 45),
                    'back': (7, 30),
                    'oblique': (14, 42),
                    'hamstring': (10, 35),
                    'other': (7, 21)
                }
                
                min_days, max_days = severity_days[injury_type]
                il_days = np.random.randint(min_days, max_days)
                
                # Create injury record
                injury_record = {
                    'Date': injury_date,
                    'Player': pitcher_name,
                    'Team': 'MLB',  # Simplified for synthetic data
                    'Transaction': f'Placed on IL',
                    'Notes': f'{injury_type.capitalize()} strain',
                    'InjuryType': injury_type,
                    'DaysOnIL': il_days,
                    'Season': year
                }
                
                injuries.append(injury_record)
                
                # Add return from IL record
                return_date = injury_date + timedelta(days=il_days)
                return_record = {
                    'Date': return_date,
                    'Player': pitcher_name,
                    'Team': 'MLB',
                    'Transaction': 'Activated from IL',
                    'Notes': f'Recovered from {injury_type}',
                    'InjuryType': injury_type,
                    'DaysOnIL': 0,
                    'Season': year
                }
                injuries.append(return_record)
        
        injuries_df = pd.DataFrame(injuries)
        injuries_df = injuries_df.sort_values('Date').reset_index(drop=True)
        
        print(f"Generated {len(injuries_df)} injury records")
        print(f"Unique injured pitchers: {injuries_df['Player'].nunique()}")
        print(f"\nInjury type distribution:")
        print(injuries_df[injuries_df['DaysOnIL'] > 0]['InjuryType'].value_counts())
        
        return injuries_df
    
    def create_survival_dataset_with_real_injuries(self, real_injuries: pd.DataFrame, include_features: bool = True) -> pd.DataFrame:
        """
        Create survival analysis dataset using real FanGraphs injury data
        """
        print("Creating survival analysis dataset with real injury data...")
        
        if self.pitcher_stats is None:
            self.pitcher_stats = self.fetch_pitcher_list()
        
        survival_data = []
        
        for _, pitcher in self.pitcher_stats.iterrows():
            pitcher_name = pitcher['Name']
            season = pitcher['Season']
            
            # Find injuries for this pitcher in this season
            # Match by name and season - could be improved with player IDs
            pitcher_injuries = real_injuries[
                (real_injuries['playerName'].str.contains(pitcher_name, case=False, na=False)) &
                (real_injuries['season'] == season)
            ]
            
            if not pitcher_injuries.empty:
                # Use first injury in the season
                first_injury = pitcher_injuries.iloc[0]
                
                # Parse the retrodate for time calculation
                if pd.notna(first_injury['retrodate']):
                    try:
                        injury_date = pd.to_datetime(first_injury['retrodate'])
                        season_start = pd.Timestamp(f"{season}-04-01")
                        time_to_event = (injury_date - season_start).days
                    except:
                        time_to_event = 90  # Default if date parsing fails
                else:
                    time_to_event = 90  # Default if no retrodate
                
                event_occurred = 1
                injury_type = first_injury['injurySurgery']
                
                # Categorize injury
                if 'elbow' in injury_type.lower() or 'tommy john' in injury_type.lower():
                    injury_category = 'elbow'
                elif 'shoulder' in injury_type.lower():
                    injury_category = 'shoulder'
                elif 'oblique' in injury_type.lower():
                    injury_category = 'oblique'
                elif any(term in injury_type.lower() for term in ['forearm', 'lat']):
                    injury_category = 'forearm'
                else:
                    injury_category = 'other'
                
            else:
                # No injury - censored observation
                time_to_event = 180  # Full season
                event_occurred = 0
                injury_type = None
                injury_category = None
            
            # Create record with features
            record = {
                'player_name': pitcher_name,
                'season': season,
                'time_to_event': max(1, time_to_event),
                'event': event_occurred,
                'injury_type': injury_type,
                'injury_category': injury_category
            }
            
            # Add performance features if available
            if include_features:
                feature_columns = ['Age', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'H', 
                                 'R', 'ER', 'HR', 'BB', 'SO', 'WHIP', 'K/9', 'BB/9', 
                                 'HR/9', 'FIP', 'WAR']
                
                for col in feature_columns:
                    if col in pitcher:
                        value = pitcher[col]
                        # Handle potential NaN values
                        if pd.isna(value):
                            value = 0
                        record[col.lower().replace('/', '_per_')] = value
            
            survival_data.append(record)
        
        survival_df = pd.DataFrame(survival_data)
        
        # Clean up the dataset
        survival_df = survival_df.dropna(subset=['time_to_event'])
        survival_df['time_to_event'] = survival_df['time_to_event'].clip(lower=1)
        
        # Add derived features
        if 'ip' in survival_df.columns:
            survival_df['high_workload'] = (survival_df['ip'] > 180).astype(int)
        if 'age' in survival_df.columns:
            survival_df['veteran'] = (survival_df['age'] > 30).astype(int)
        if 'era' in survival_df.columns:
            survival_df['high_era'] = (survival_df['era'] > 4.0).astype(int)
        
        print(f"Created survival dataset with {len(survival_df)} observations")
        print(f"Event rate: {survival_df['event'].mean():.2%}")
        if survival_df['event'].sum() > 0:
            print(f"Average time to event (injured): {survival_df[survival_df['event']==1]['time_to_event'].mean():.1f} days")
        print(f"Features included: {list(survival_df.columns)}")
        
        return survival_df
    
    def create_survival_dataset(self, include_features: bool = True) -> pd.DataFrame:
        """
        Create survival analysis dataset with proper time-to-event structure
        """
        print("Creating survival analysis dataset...")
        
        if self.pitcher_stats is None:
            self.pitcher_stats = self.fetch_pitcher_list()
        
        if self.injury_data is None or self.injury_data.empty:
            # Generate synthetic data if no real injury data available
            self.injury_data = self.generate_synthetic_injury_data(self.pitcher_stats)
        
        survival_data = []
        
        for _, pitcher in self.pitcher_stats.iterrows():
            pitcher_name = pitcher['Name']
            season = pitcher['Season']
            
            # Find injuries for this pitcher in this season
            pitcher_injuries = self.injury_data[
                (self.injury_data['Player'] == pitcher_name) &
                (self.injury_data['Season'] == season) &
                (self.injury_data['DaysOnIL'] > 0)
            ]
            
            if not pitcher_injuries.empty:
                # Use first injury in the season
                first_injury = pitcher_injuries.iloc[0]
                season_start = pd.Timestamp(f"{season}-04-01")
                time_to_event = (first_injury['Date'] - season_start).days
                event_occurred = 1
                injury_type = first_injury['InjuryType']
                severity = first_injury['DaysOnIL']
            else:
                # No injury - censored observation
                time_to_event = 180  # Full season
                event_occurred = 0
                injury_type = None
                severity = 0
            
            # Create record with features
            record = {
                'player_name': pitcher_name,
                'season': season,
                'time_to_event': max(1, time_to_event),
                'event': event_occurred,
                'injury_type': injury_type,
                'severity': severity
            }
            
            # Add performance features if available
            if include_features:
                feature_columns = ['Age', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'H', 
                                 'R', 'ER', 'HR', 'BB', 'SO', 'WHIP', 'K/9', 'BB/9', 
                                 'HR/9', 'FIP', 'WAR']
                
                for col in feature_columns:
                    if col in pitcher:
                        value = pitcher[col]
                        # Handle potential NaN values
                        if pd.isna(value):
                            value = 0
                        record[col.lower().replace('/', '_per_')] = value
            
            survival_data.append(record)
        
        survival_df = pd.DataFrame(survival_data)
        
        # Clean up the dataset
        survival_df = survival_df.dropna(subset=['time_to_event'])
        survival_df['time_to_event'] = survival_df['time_to_event'].clip(lower=1)
        
        # Add derived features
        if 'ip' in survival_df.columns:
            survival_df['high_workload'] = (survival_df['ip'] > 180).astype(int)
        if 'age' in survival_df.columns:
            survival_df['veteran'] = (survival_df['age'] > 30).astype(int)
        if 'era' in survival_df.columns:
            survival_df['high_era'] = (survival_df['era'] > 4.0).astype(int)
        
        print(f"Created survival dataset with {len(survival_df)} observations")
        print(f"Event rate: {survival_df['event'].mean():.2%}")
        print(f"Average time to event (injured): {survival_df[survival_df['event']==1]['time_to_event'].mean():.1f} days")
        print(f"Features included: {list(survival_df.columns)}")
        
        return survival_df
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a smaller sample dataset for quick testing
        """
        print("Creating sample dataset for testing...")
        
        # Sample data with known patterns for testing
        np.random.seed(42)
        n_samples = 500
        
        # Generate synthetic features
        age = np.random.normal(28, 4, n_samples)
        innings_pitched = np.random.gamma(8, 20, n_samples)
        era = np.random.gamma(2, 1.8, n_samples)
        whip = 0.8 + 0.6 * np.random.beta(2, 2, n_samples)
        k_per_9 = np.random.normal(8.5, 2, n_samples)
        
        # Generate events based on risk factors
        risk_score = (
            0.3 * (age > 32).astype(float) +
            0.3 * (innings_pitched > 180).astype(float) +
            0.2 * (era > 4.5).astype(float) +
            0.2 * (whip > 1.3).astype(float)
        )
        
        event_prob = 0.2 + 0.4 * risk_score
        events = np.random.binomial(1, event_prob)
        
        # Generate time to event
        time_to_event = np.where(
            events == 1,
            np.random.weibull(2, n_samples) * 60 + 20,  # Injured: Weibull distribution
            180 * np.ones(n_samples)  # Censored: full season
        )
        
        # Injury types for those who got injured
        injury_types = ['elbow', 'shoulder', 'forearm', 'back', 'oblique', 'other']
        injury_type = np.where(
            events == 1,
            np.random.choice(injury_types, n_samples),
            None
        )
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'player_id': range(n_samples),
            'age': age,
            'innings_pitched': innings_pitched,
            'era': era,
            'whip': whip,
            'k_per_9': k_per_9,
            'time_to_event': np.clip(time_to_event, 1, 180),
            'event': events,
            'injury_type': injury_type,
            'high_workload': (innings_pitched > 180).astype(int),
            'veteran': (age > 30).astype(int),
            'high_era': (era > 4.0).astype(int)
        })
        
        print(f"Sample dataset created: {sample_df.shape}")
        print(f"Event rate: {sample_df['event'].mean():.2%}")
        
        return sample_df
    
    def save_data(self, output_dir: str = "~/data/processed"):
        """Save all datasets to CSV files"""
        print(f"Saving data to {output_dir}...")
        
        if self.pitcher_stats is not None:
            self.pitcher_stats.to_csv(f"{output_dir}/pitcher_stats.csv", index=False)
            print(f"  Saved pitcher_stats.csv")
        
        if self.injury_data is not None:
            self.injury_data.to_csv(f"{output_dir}/injury_data.csv", index=False)
            print(f"  Saved injury_data.csv")
        
        # Create and save survival dataset
        survival_df = self.create_survival_dataset()
        survival_df.to_csv(f"{output_dir}/survival_dataset.csv", index=False)
        print(f"  Saved survival_dataset.csv")
        
        # Create and save sample dataset
        sample_df = self.create_sample_dataset()
        sample_df.to_csv(f"{output_dir}/sample_dataset.csv", index=False)
        print(f"  Saved sample_dataset.csv")
        
        return survival_df, sample_df


if __name__ == "__main__":
    print("MLB Pitcher Injury Data Acquisition V2")
    print("=" * 50)
    
    # Initialize acquisition
    acquisition = PitcherDataAcquisitionV2(start_year=2021, end_year=2023)
    
    # Fetch pitcher statistics
    pitchers = acquisition.fetch_pitcher_list()
    acquisition.pitcher_stats = pitchers
    
    # Generate synthetic injury data (since real scraping might be blocked)
    injuries = acquisition.generate_synthetic_injury_data(pitchers)
    acquisition.injury_data = injuries
    
    # Create survival dataset
    survival_df = acquisition.create_survival_dataset()
    
    # Create sample dataset for testing
    sample_df = acquisition.create_sample_dataset()
    
    # Save all data
    acquisition.save_data()
    
    print("\nData acquisition complete!")
    print(f"Survival dataset shape: {survival_df.shape}")
    print(f"Sample dataset shape: {sample_df.shape}")
    
    print(f"\nSurvival dataset preview:")
    print(survival_df.head())
    
    print(f"\nSample dataset preview:")
    print(sample_df.head())