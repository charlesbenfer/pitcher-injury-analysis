"""
Create proper team mapping using real MLB data
"""

import pandas as pd
import numpy as np

def create_team_mapping():
    print("Creating real team mapping from MLB data...")
    
    # Load injury data (has player_id -> team mapping)
    print("Loading injury data...")
    injury_data = pd.read_csv('data/raw/pitcher_injuries_2018_2024.csv')
    
    # Load pitcher data (has player_id -> full_name mapping)
    print("Loading pitcher data...")
    pitcher_data = pd.read_csv('data/raw/pitchers_2018_2024.csv')
    
    # Create player_id -> team mapping from injury data
    # Use the most recent team for each player
    player_team_mapping = injury_data.groupby('player_id').agg({
        'team_name': 'last',
        'team': 'last',
        'year': 'max'
    }).reset_index()
    
    print(f"Found {len(player_team_mapping)} players with team information")
    
    # Create player_id -> name mapping from pitcher data
    player_name_mapping = pitcher_data.groupby('player_id').agg({
        'full_name': 'first'
    }).reset_index()
    
    print(f"Found {len(player_name_mapping)} players with name information")
    
    # Merge to get player_name -> team mapping
    team_mapping = pd.merge(player_name_mapping, player_team_mapping, on='player_id', how='inner')
    
    print(f"Successfully matched {len(team_mapping)} players with both names and teams")
    
    # Create final mapping dictionary
    name_to_team = dict(zip(team_mapping['full_name'], team_mapping['team_name']))
    
    # Load survival dataset to check coverage
    print("Checking coverage against survival dataset...")
    survival_data = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
    
    # Check how many players we can map
    survival_players = set(survival_data['player_name'].unique())
    mapped_players = set(name_to_team.keys())
    
    print(f"Survival dataset has {len(survival_players)} unique players")
    print(f"Team mapping covers {len(mapped_players)} players")
    print(f"Overlap: {len(survival_players & mapped_players)} players ({len(survival_players & mapped_players)/len(survival_players)*100:.1f}%)")
    
    # Show some unmapped players
    unmapped = survival_players - mapped_players
    if unmapped:
        print(f"\nFirst 10 unmapped players: {list(unmapped)[:10]}")
    
    # Save the mapping
    mapping_df = pd.DataFrame([
        {'player_name': name, 'team_name': team} 
        for name, team in name_to_team.items()
    ])
    
    mapping_df.to_csv('data/processed/player_team_mapping.csv', index=False)
    print(f"\nSaved team mapping to data/processed/player_team_mapping.csv")
    
    return name_to_team

if __name__ == "__main__":
    mapping = create_team_mapping()