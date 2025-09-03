"""
Regenerate 2025 projections with accurate team assignments
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import json
from datetime import datetime

print("🎯 Regenerating 2025 Projections with Accurate Team Assignments...")

# Load full dataset
df = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
print(f"Loaded {len(df)} pitcher-seasons from 2019-2024")

# Add accurate team data
print(f"\n🏟️ Loading accurate team assignments...")
try:
    # Use the current accurate mapping
    team_mapping = pd.read_csv('data/processed/player_team_mapping_current.csv')
    # Handle 'Multiple Teams' as 'Traded'  
    team_mapping['team_name'] = team_mapping['team_name'].replace('Multiple Teams', 'Traded')
    name_to_team = dict(zip(team_mapping['player_name'], team_mapping['team_name']))
    df['team'] = df['player_name'].map(name_to_team)
    
    # Handle unmapped players
    unmapped_count = df['team'].isna().sum()
    df['team'] = df['team'].fillna('Unknown Team')
    
    print(f"✅ Team mapping coverage: {(len(df) - unmapped_count)/len(df)*100:.1f}%")
    print(f"❓ Unmapped players: {unmapped_count}")
    
    # Show some team assignments for verification
    print(f"\n📋 Sample team assignments:")
    key_players = ['Michael King', 'Carlos Rodon', 'Gerrit Cole', 'Dylan Cease', 'Blake Snell']
    for player in key_players:
        if player in df['player_name'].values:
            player_team = df[df['player_name'] == player]['team'].iloc[0]
            print(f"  • {player}: {player_team}")
            
except FileNotFoundError:
    print("❌ Current team mapping not found")
    df['team'] = 'Unknown Team'

# Split data chronologically
train_data = df[df['season'] < 2024].copy()
holdout_2024 = df[df['season'] == 2024].copy()

print(f"\n📊 Data Split:")
print(f"Training data (2019-2023): {len(train_data)} observations")
print(f"Holdout data (2024): {len(holdout_2024)} observations")

# Use the existing risk scorer approach
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

# Create 2025 projections
print(f"\n⚾ Creating 2025 Season Projections with Accurate Teams...")

# Use 2024 stats as lagged features for 2025 predictions
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
        'team': pitcher.get('team', 'Unknown Team'),  # Use 2024 team assignment
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

# Verify key players have correct teams
print(f"\n✅ Verifying team assignments in 2025 projections:")
key_players_check = ['Michael King', 'Carlos Rodon', 'Gerrit Cole', 'Dylan Cease']
for player in key_players_check:
    if player in projections_2025_df['player_name'].values:
        player_data = projections_2025_df[projections_2025_df['player_name'] == player].iloc[0]
        print(f"  • {player}: {player_data['team']} ({player_data['risk_category_2025']} Risk)")

# 2025 Risk distribution
print(f"\n📈 2025 Season Risk Distribution:")
risk_2025_counts = projections_2025_df['risk_category_2025'].value_counts()
for category in ['Low', 'Moderate', 'High', 'Very High']:
    count = risk_2025_counts.get(category, 0)
    pct = count / len(projections_2025_df) * 100
    print(f"  {category}: {count} pitchers ({pct:.1f}%)")

# Team distribution in 2025
print(f"\n🏟️ Team Distribution (2025):")
team_counts = projections_2025_df['team'].value_counts()
for team, count in team_counts.head(10).items():
    print(f"  {team}: {count} players")

# Save results
print(f"\n💾 Saving Results...")

# Save 2025 projections with accurate teams
projections_2025_df.to_csv('data/processed/pitcher_projections_2025.csv', index=False)
print(f"Saved 2025 projections: data/processed/pitcher_projections_2025.csv")

print(f"\n✅ Projections Complete with Accurate Team Assignments!")
print(f"⚾ 2025 projections ready for {len(projections_2025_df)} pitchers")

# Show teams with most high-risk pitchers
high_risk_2025 = projections_2025_df[projections_2025_df['alert_level_2025'] >= 2]
if len(high_risk_2025) > 0:
    team_risk_counts = high_risk_2025['team'].value_counts()
    print(f"\n🚨 Teams with Most High-Risk Pitchers (2025):")
    for team, count in team_risk_counts.head(5).items():
        print(f"  {team}: {count} high-risk pitchers")