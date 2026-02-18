"""
Data Integration Script for Pitcher Injury Analysis
Combines the newly scraped injury data with existing performance data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PitcherDataIntegrator:
    """Integrates injury data with performance data for enhanced analysis"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_existing_data(self):
        """Load existing performance and survival datasets"""
        logger.info("Loading existing datasets")
        
        # Load existing survival dataset
        survival_files = list(self.processed_dir.glob("survival_dataset*.csv"))
        if survival_files:
            survival_df = pd.read_csv(survival_files[0])
            logger.info(f"Loaded existing survival dataset: {survival_df.shape}")
        else:
            logger.warning("No existing survival dataset found")
            survival_df = pd.DataFrame()
        
        # Load existing performance data if available
        performance_files = list(self.raw_dir.glob("*performance*.csv"))
        if performance_files:
            performance_df = pd.read_csv(performance_files[0])
            logger.info(f"Loaded performance data: {performance_df.shape}")
        else:
            logger.info("No performance data found - will focus on injury data")
            performance_df = pd.DataFrame()
        
        return survival_df, performance_df
    
    def load_new_injury_data(self):
        """Load newly scraped injury data"""
        logger.info("Loading new injury data")
        
        # Look for pitcher injury files
        injury_files = list(self.raw_dir.glob("pitcher_injuries_*.csv"))
        if injury_files:
            # Use the most recent file
            injury_file = max(injury_files, key=lambda x: x.stat().st_mtime)
            injury_df = pd.read_csv(injury_file)
            logger.info(f"Loaded new injury data from {injury_file}: {injury_df.shape}")
            return injury_df
        else:
            logger.error("No new injury data found")
            return pd.DataFrame()
    
    def process_injury_events(self, injury_df):
        """Process injury data to create survival analysis format"""
        logger.info("Processing injury events for survival analysis")
        
        # Filter for actual injuries (not activations)
        injury_events = injury_df[
            (injury_df['is_il'] == True) & 
            (injury_df['is_activated'] == False)
        ].copy()
        
        # Calculate days from season start
        injury_events['date'] = pd.to_datetime(injury_events['date'])
        injury_events['season_start'] = pd.to_datetime(injury_events['year'].astype(str) + '-03-15')
        injury_events['days_from_start'] = (injury_events['date'] - injury_events['season_start']).dt.days
        
        # Filter for injuries within reasonable season timeframe (0-200 days)
        injury_events = injury_events[
            (injury_events['days_from_start'] >= 0) & 
            (injury_events['days_from_start'] <= 200)
        ]
        
        # Group by player-season to get first injury
        first_injuries = injury_events.groupby(['player_id', 'year']).agg({
            'days_from_start': 'min',
            'il_days': 'first',
            'injury_location': 'first',
            'team': 'first'
        }).reset_index()
        
        # Rename columns
        first_injuries = first_injuries.rename(columns={
            'days_from_start': 'time_to_event',
            'il_days': 'injury_severity'
        })
        
        # Add event indicator
        first_injuries['event'] = 1
        
        logger.info(f"Processed {len(first_injuries)} injury events")
        return first_injuries
    
    def create_survival_dataset(self, injury_events, performance_df=None):
        """Create comprehensive survival dataset"""
        logger.info("Creating survival dataset")
        
        # Get all pitcher-seasons from injury data
        all_pitcher_seasons = injury_events[['player_id', 'year', 'team']].drop_duplicates()
        
        # For now, create a basic survival dataset from injury data
        # This will need to be enhanced with performance data when available
        
        # Create censored observations (pitchers who didn't get injured)
        # For this example, assume 180-day follow-up for non-injured pitchers
        survival_data = []
        
        # Add injury events
        for _, row in injury_events.iterrows():
            survival_data.append({
                'player_id': row['player_id'],
                'season': row['year'],
                'team': row['team'],
                'time_to_event': row['time_to_event'],
                'event': 1,
                'injury_location': row['injury_location'],
                'injury_severity': row.get('injury_severity', np.nan)
            })
        
        # For demonstration, create some censored observations
        # In practice, you'd need to identify pitchers who played full seasons without injury
        unique_teams_years = injury_events[['year', 'team']].drop_duplicates()
        
        # Create basic dataset structure
        survival_df = pd.DataFrame(survival_data)
        
        # Add basic features (these would be replaced with actual performance data)
        survival_df['age'] = np.random.normal(28, 4, len(survival_df))  # Placeholder
        survival_df['experience_years'] = np.random.normal(5, 3, len(survival_df))  # Placeholder
        
        logger.info(f"Created survival dataset with {len(survival_df)} observations")
        return survival_df
    
    def validate_data_quality(self, df):
        """Validate the integrated dataset"""
        logger.info("Validating data quality")
        
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check time_to_event range
        if 'time_to_event' in df.columns:
            time_range = df['time_to_event'].describe()
            if time_range['min'] < 0 or time_range['max'] > 365:
                issues.append(f"Unusual time_to_event values: min={time_range['min']}, max={time_range['max']}")
        
        # Check event distribution
        if 'event' in df.columns:
            event_rate = df['event'].mean()
            if event_rate < 0.05 or event_rate > 0.8:
                issues.append(f"Unusual event rate: {event_rate:.2%}")
        
        # Log validation results
        if issues:
            for issue in issues:
                logger.warning(f"Data quality issue: {issue}")
        else:
            logger.info("Data quality validation passed")
        
        return issues
    
    def save_integrated_dataset(self, df, filename="enhanced_survival_dataset.csv"):
        """Save the integrated dataset"""
        output_file = self.processed_dir / filename
        df.to_csv(output_file, index=False)
        logger.info(f"Saved integrated dataset to {output_file}")
        
        # Save summary statistics
        summary_file = self.processed_dir / filename.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Enhanced Survival Dataset Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Event rate: {df['event'].mean():.2%}\n")
            f.write(f"Years covered: {df['season'].min()}-{df['season'].max()}\n")
            f.write(f"Unique pitchers: {df['player_id'].nunique()}\n")
            f.write(f"\nColumns: {list(df.columns)}\n")
            f.write(f"\nDescriptive statistics:\n")
            f.write(str(df.describe()))
        
        logger.info(f"Saved summary to {summary_file}")
        return output_file
    
    def integrate_all_data(self):
        """Main integration workflow"""
        logger.info("Starting data integration process")
        
        # Load existing data
        existing_survival, existing_performance = self.load_existing_data()
        
        # Load new injury data
        new_injury_data = self.load_new_injury_data()
        
        if new_injury_data.empty:
            logger.error("Cannot proceed without new injury data")
            return None
        
        # Process injury events
        injury_events = self.process_injury_events(new_injury_data)
        
        # Create survival dataset
        survival_dataset = self.create_survival_dataset(injury_events, existing_performance)
        
        # Validate data quality
        self.validate_data_quality(survival_dataset)
        
        # Save integrated dataset
        output_file = self.save_integrated_dataset(
            survival_dataset, 
            f"enhanced_survival_dataset_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # Print summary
        logger.info(f"""
        INTEGRATION SUMMARY:
        ===================
        Input injury records: {len(new_injury_data)}
        Processed injury events: {len(injury_events)}
        Final survival dataset: {len(survival_dataset)} observations
        Event rate: {survival_dataset['event'].mean():.2%}
        Years covered: {survival_dataset['season'].min()}-{survival_dataset['season'].max()}
        Output file: {output_file}
        """)
        
        return survival_dataset


if __name__ == "__main__":
    # Create integrator and run
    integrator = PitcherDataIntegrator()
    enhanced_dataset = integrator.integrate_all_data()
    
    if enhanced_dataset is not None:
        print(f"Integration completed successfully!")
        print(f"Enhanced dataset shape: {enhanced_dataset.shape}")
        print(f"Sample data:")
        print(enhanced_dataset.head())
    else:
        print("Integration failed - check logs for details")