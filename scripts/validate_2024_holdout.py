"""
2024 Holdout Validation and 2025 Projections
External validation on unseen 2024 data and forward projections.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import json
from datetime import datetime

print("🎯 Starting 2024 Holdout Validation and 2025 Projections...")

# Load full dataset
df = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
print(f"Loaded {len(df)} pitcher-seasons from 2019-2024")
print(f"Season distribution:\n{df['season'].value_counts().sort_index()}")

# Add team data to the dataset
print(f"\n🏟️ Adding team assignments...")
try:
    team_mapping = pd.read_csv('data/processed/player_team_mapping.csv')
    name_to_team = dict(zip(team_mapping['player_name'], team_mapping['team_name']))
    df['team'] = df['player_name'].map(name_to_team)
    
    # Handle unmapped players
    unmapped_count = df['team'].isna().sum()
    df['team'] = df['team'].fillna('Unknown Team')
    
    print(f"Team mapping coverage: {(len(df) - unmapped_count)/len(df)*100:.1f}%")
    print(f"Unmapped players (likely traded): {unmapped_count}")
except FileNotFoundError:
    print("Warning: Team mapping file not found. Creating placeholder team assignments.")
    df['team'] = 'Unknown Team'

# Split data chronologically
train_data = df[df['season'] < 2024].copy()
holdout_2024 = df[df['season'] == 2024].copy()

print(f"\n📊 Data Split:")
print(f"Training data (2019-2023): {len(train_data)} observations")
print(f"Holdout data (2024): {len(holdout_2024)} observations")
print(f"Training injury rate: {train_data['event'].mean():.1%}")
print(f"2024 injury rate: {holdout_2024['event'].mean():.1%}")

# Use the existing risk scorer approach for consistency
class PitcherRiskScorer:
    """Production-ready pitcher injury risk scoring system"""
    
    def __init__(self):
        # Use the same calibrated weights from the dashboard
        self.risk_quartiles = [-5.154, -5.008, -4.931]
        self.beta_0 = 5.0
        self.alpha = 2.0
        
        # Feature weights (reduced by 50% for calibration)
        self.feature_weights = {
            'age_prev': -0.01,
            'g_prev': 0.0075,
            'veteran_prev': 0.004,
            'era_prev': -0.004,
            'ip_prev': 0.0004,
            'war_prev': 0.002,
            'high_workload_prev': -0.002
        }
    
    def calculate_risk_score(self, pitcher_stats):
        """Calculate comprehensive risk assessment for a pitcher"""
        # Calculate linear predictor
        linear_pred = self.beta_0
        for feature, weight in self.feature_weights.items():
            if feature in pitcher_stats:
                linear_pred += weight * pitcher_stats[feature]
        
        risk_score = -linear_pred
        
        # Determine risk category
        if risk_score <= self.risk_quartiles[0]:
            risk_category = 'Low'
            alert_level = 0
            color_code = '🟢'
            color_hex = '#2E8B57'
        elif risk_score <= self.risk_quartiles[1]:
            risk_category = 'Moderate'
            alert_level = 1
            color_code = '🟡'
            color_hex = '#FFD700'
        elif risk_score <= self.risk_quartiles[2]:
            risk_category = 'High'
            alert_level = 2
            color_code = '🟠'
            color_hex = '#FF8C00'
        else:
            risk_category = 'Very High'
            alert_level = 3
            color_code = '🔴'
            color_hex = '#DC143C'
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'alert_level': alert_level,
            'color_code': color_code,
            'color_hex': color_hex
        }

# Initialize scorer
risk_scorer = PitcherRiskScorer()

# Features for modeling
features = [
    'age_prev', 'g_prev', 'veteran_prev', 'era_prev', 
    'ip_prev', 'war_prev', 'high_workload_prev'
]

print(f"\n🔍 Calculating risk scores for 2024 holdout data...")

# Calculate risk scores for 2024 holdout
holdout_results = []
for _, pitcher in holdout_2024.iterrows():
    pitcher_stats = pitcher[features].to_dict()
    risk_result = risk_scorer.calculate_risk_score(pitcher_stats)
    
    holdout_results.append({
        'player_name': pitcher['player_name'],
        'season': pitcher['season'],
        'actual_injury': pitcher['event'],
        'time_to_event': pitcher['time_to_event'],
        'risk_score': risk_result['risk_score'],
        'risk_category': risk_result['risk_category'],
        'alert_level': risk_result['alert_level'],
        **{f: pitcher[f] for f in features}
    })

holdout_df = pd.DataFrame(holdout_results)

# Calculate external validation metrics
risk_scores_2024 = holdout_df['risk_score'].values
y_time_holdout = holdout_df['time_to_event'].values
y_event_holdout = holdout_df['actual_injury'].values

# C-index calculation (higher risk score should predict shorter survival time)
c_index_2024 = concordance_index(y_time_holdout, -risk_scores_2024, y_event_holdout)

print(f"\n🎯 2024 Holdout Validation Results:")
print(f"C-index: {c_index_2024:.3f}")
print(f"Injuries in 2024: {y_event_holdout.sum()}/{len(y_event_holdout)} ({y_event_holdout.mean():.1%})")

# Risk calibration check
print(f"\n📊 Risk Calibration on 2024 Data:")
calibration_results = holdout_df.groupby('risk_category').agg({
    'actual_injury': ['count', 'sum', 'mean'],
    'time_to_event': 'mean',
    'alert_level': 'first'
}).round(3)

calibration_results.columns = ['Count', 'Injuries', 'Injury_Rate', 'Avg_Time', 'Alert_Level']
calibration_results = calibration_results.sort_values('Alert_Level')
print(calibration_results)

# Check if injury rates increase with risk level (monotonic)
injury_rates = calibration_results['Injury_Rate'].values
is_monotonic = all(injury_rates[i] <= injury_rates[i+1] for i in range(len(injury_rates)-1))
print(f"Monotonic risk gradient: {'✅ Yes' if is_monotonic else '❌ No'}")

# Create 2025 projections
print(f"\n⚾ Creating 2025 Season Projections...")

# Use 2024 stats as lagged features for 2025 predictions
projections_2025 = holdout_2024.copy()
projections_2025['season'] = 2025

# Create 2025 projection features (2024 becomes "prev" for 2025)
projection_data = []
for _, pitcher in holdout_2024.iterrows():
    # Age up by 1 year for 2025
    age_2025 = pitcher['age'] + 1
    
    projection_stats = {
        'age_prev': age_2025,
        'g_prev': pitcher['g'],
        'veteran_prev': int(age_2025 >= 30),
        'era_prev': pitcher['era'],
        'ip_prev': pitcher['ip'],
        'war_prev': pitcher['war'],
        'high_workload_prev': pitcher['high_workload']
    }
    
    # Calculate 2025 risk prediction
    risk_result_2025 = risk_scorer.calculate_risk_score(projection_stats)
    
    projection_data.append({
        'player_name': pitcher['player_name'],
        'season': 2025,
        'age': age_2025,
        'team': pitcher.get('team', 'Unknown Team'),  # Use 2024 team assignment for 2025 projections
        'risk_score_2025': risk_result_2025['risk_score'],
        'risk_category_2025': risk_result_2025['risk_category'],
        'alert_level_2025': risk_result_2025['alert_level'],
        'color_code_2025': risk_result_2025['color_code'],
        'color_hex_2025': risk_result_2025['color_hex'],
        # Include 2024 stats for reference
        'g_2024': pitcher['g'],
        'era_2024': pitcher['era'],
        'ip_2024': pitcher['ip'],
        'war_2024': pitcher['war'],
        **projection_stats
    })

projections_2025_df = pd.DataFrame(projection_data)

print(f"Generated 2025 projections for {len(projections_2025_df)} active pitchers")

# 2025 Risk distribution
print(f"\n📈 2025 Season Risk Distribution:")
risk_2025_counts = projections_2025_df['risk_category_2025'].value_counts()
for category in ['Low', 'Moderate', 'High', 'Very High']:
    count = risk_2025_counts.get(category, 0)
    pct = count / len(projections_2025_df) * 100
    print(f"  {category}: {count} pitchers ({pct:.1f}%)")

# High-risk pitchers for 2025
high_risk_2025 = projections_2025_df[projections_2025_df['alert_level_2025'] >= 2]
print(f"\n🚨 High-Risk Pitchers for 2025: {len(high_risk_2025)}")
if len(high_risk_2025) > 0:
    print("Top 10 highest risk:")
    top_risk = high_risk_2025.nlargest(10, 'risk_score_2025')[['player_name', 'risk_category_2025', 'age', 'era_2024']]
    for _, pitcher in top_risk.iterrows():
        print(f"  • {pitcher['player_name']} ({pitcher['age']}) - {pitcher['risk_category_2025']} Risk (ERA: {pitcher['era_2024']:.2f})")

# Save results
print(f"\n💾 Saving Results...")

# Save 2025 projections
projections_2025_df.to_csv('data/processed/pitcher_projections_2025.csv', index=False)
print(f"Saved 2025 projections: data/processed/pitcher_projections_2025.csv")

# Save holdout validation results
holdout_df.to_csv('data/processed/holdout_validation_2024.csv', index=False)
print(f"Saved validation results: data/processed/holdout_validation_2024.csv")

# Save validation summary
validation_summary = {
    'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_period': '2019-2023',
    'holdout_period': '2024',
    'training_samples': len(train_data),
    'holdout_samples': len(holdout_2024),
    'c_index_2024': float(c_index_2024),
    'training_injury_rate': float(train_data['event'].mean()),
    'holdout_injury_rate': float(holdout_2024['event'].mean()),
    'projection_samples_2025': len(projections_2025_df),
    'monotonic_risk_gradient': is_monotonic,
    'high_risk_2025_count': len(high_risk_2025),
    'risk_distribution_2025': {
        'Low': int(risk_2025_counts.get('Low', 0)),
        'Moderate': int(risk_2025_counts.get('Moderate', 0)),
        'High': int(risk_2025_counts.get('High', 0)),
        'Very High': int(risk_2025_counts.get('Very High', 0))
    }
}

with open('data/processed/validation_summary.json', 'w') as f:
    json.dump(validation_summary, f, indent=2)

print(f"Saved validation summary: data/processed/validation_summary.json")

print(f"\n✅ Validation and Projections Complete!")
print(f"📊 External validation C-index: {c_index_2024:.3f}")
print(f"⚾ 2025 projections ready for {len(projections_2025_df)} pitchers")
print(f"🚨 {len(high_risk_2025)} high-risk pitchers identified for 2025")