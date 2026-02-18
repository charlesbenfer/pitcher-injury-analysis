"""
Fetch current MLB team assignments for all pitchers
Uses pybaseball to get accurate, up-to-date roster information
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔍 Fetching current MLB team assignments...")

try:
    from pybaseball import playerid_lookup, team_pitching, pitching_stats
    print("✅ Using pybaseball for current roster data")
except ImportError:
    print("📦 Installing pybaseball for MLB data access...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pybaseball'])
    from pybaseball import playerid_lookup, team_pitching, pitching_stats

def get_2024_team_rosters():
    """
    Fetch 2024 team rosters from Baseball Reference using pybaseball
    """
    print("\n⚾ Fetching 2024 MLB pitcher data...")
    
    # Get 2024 pitching stats with team information
    # This includes all pitchers who played in 2024
    try:
        # Get all pitching stats for 2024 season
        pitching_2024 = pitching_stats(2024, qual=0)  # qual=0 gets all pitchers
        
        print(f"Found {len(pitching_2024)} pitchers in 2024 season")
        
        # Create clean mapping of player to team
        # Handle players who played for multiple teams
        team_mapping = {}
        
        for _, row in pitching_2024.iterrows():
            player_name = row['Name']
            team = row['Team']
            
            # Handle players on multiple teams
            if team == 'TOT':
                # Skip total rows for now
                continue
            
            # Clean up team names
            if team == '- - -' or team == '---' or pd.isna(team):
                # Mark as traded/multiple teams
                team = 'Multiple Teams'
            
            # Store the team for this player
            # If player already exists, keep the last team (most recent)
            team_mapping[player_name] = team
        
        print(f"Successfully mapped {len(team_mapping)} players to teams")
        
        return team_mapping
        
    except Exception as e:
        print(f"Error fetching from pybaseball: {e}")
        print("Attempting alternative approach...")
        return None

def get_fangraphs_teams():
    """
    Alternative: Use FanGraphs data via pybaseball
    """
    print("\n📊 Trying FanGraphs data source...")
    
    try:
        from pybaseball import pitching_stats_range
        
        # Get 2024 season pitching data from FanGraphs
        # This includes team information
        stats_2024 = pitching_stats_range('2024-04-01', '2024-10-01')
        
        team_mapping = {}
        for _, row in stats_2024.iterrows():
            if 'Name' in row and 'Team' in row:
                team_mapping[row['Name']] = row['Team']
        
        print(f"FanGraphs: Mapped {len(team_mapping)} players")
        return team_mapping
        
    except Exception as e:
        print(f"FanGraphs error: {e}")
        return None

def normalize_team_names(team_abbr):
    """
    Convert team abbreviations to full names
    """
    team_map = {
        'ARI': 'D-backs', 'AZ': 'D-backs',
        'ATL': 'Braves',
        'BAL': 'Orioles', 'BLT': 'Orioles',
        'BOS': 'Red Sox', 'BSN': 'Red Sox',
        'CHC': 'Cubs', 'CHN': 'Cubs',
        'CHW': 'White Sox', 'CWS': 'White Sox', 'CHA': 'White Sox',
        'CIN': 'Reds',
        'CLE': 'Guardians', 'CLV': 'Guardians',
        'COL': 'Rockies',
        'DET': 'Tigers',
        'HOU': 'Astros',
        'KC': 'Royals', 'KCR': 'Royals', 'KCA': 'Royals',
        'LAA': 'Angels', 'ANA': 'Angels',
        'LAD': 'Dodgers', 'LA': 'Dodgers',
        'MIA': 'Marlins', 'FLA': 'Marlins',
        'MIL': 'Brewers',
        'MIN': 'Twins',
        'NYM': 'Mets', 'NYN': 'Mets',
        'NYY': 'Yankees', 'NYA': 'Yankees',
        'OAK': 'Athletics',
        'PHI': 'Phillies',
        'PIT': 'Pirates',
        'SD': 'Padres', 'SDP': 'Padres',
        'SF': 'Giants', 'SFG': 'Giants', 'SFN': 'Giants',
        'SEA': 'Mariners',
        'STL': 'Cardinals', 'SLN': 'Cardinals',
        'TB': 'Rays', 'TBR': 'Rays', 'TBD': 'Rays',
        'TEX': 'Rangers',
        'TOR': 'Blue Jays',
        'WSH': 'Nationals', 'WAS': 'Nationals', 'WSN': 'Nationals'
    }
    
    if team_abbr in team_map:
        return team_map[team_abbr]
    return team_abbr

# Main execution
print("\n🏟️ Starting team roster fetch...")

# Try primary method
team_mapping = get_2024_team_rosters()

# Try backup method if needed
if not team_mapping:
    team_mapping = get_fangraphs_teams()

if team_mapping:
    # Convert to DataFrame
    mapping_df = pd.DataFrame([
        {'player_name': name, 'team_2024': normalize_team_names(team)}
        for name, team in team_mapping.items()
    ])
    
    # Load our existing pitcher list to match names
    print("\n🔄 Matching with existing pitcher database...")
    
    survival_data = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
    our_pitchers = survival_data['player_name'].unique()
    
    print(f"Our dataset has {len(our_pitchers)} unique pitchers")
    
    # First pass: Get players who played for multiple teams in 2024
    multi_team_players = {}
    for pitcher in our_pitchers:
        if pitcher in team_mapping and team_mapping[pitcher] == 'Multiple Teams':
            # For multi-team players, try to find their last team from the detailed data
            # For now, mark them as 'Traded'
            multi_team_players[pitcher] = 'Traded'
    
    # Try to match names (accounting for slight variations)
    matched = 0
    unmatched = []
    final_mapping = []
    
    for pitcher in our_pitchers:
        # Try exact match first
        if pitcher in team_mapping:
            final_mapping.append({
                'player_name': pitcher,
                'team_name': normalize_team_names(team_mapping[pitcher]),
                'source': 'exact_match'
            })
            matched += 1
        else:
            # Try partial match (last name)
            last_name = pitcher.split()[-1] if ' ' in pitcher else pitcher
            found = False
            for full_name, team in team_mapping.items():
                if last_name in full_name:
                    final_mapping.append({
                        'player_name': pitcher,
                        'team_name': normalize_team_names(team),
                        'source': 'partial_match'
                    })
                    matched += 1
                    found = True
                    break
            
            if not found:
                unmatched.append(pitcher)
                final_mapping.append({
                    'player_name': pitcher,
                    'team_name': 'Unknown Team',
                    'source': 'unmatched'
                })
    
    print(f"\n📊 Matching Results:")
    print(f"✅ Matched: {matched}/{len(our_pitchers)} ({matched/len(our_pitchers)*100:.1f}%)")
    print(f"❓ Unmatched: {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 20:
        print(f"\nUnmatched players: {unmatched[:20]}")
    
    # Save the updated mapping
    final_df = pd.DataFrame(final_mapping)
    final_df.to_csv('data/processed/player_team_mapping_current.csv', index=False)
    
    print(f"\n💾 Saved updated team mapping to data/processed/player_team_mapping_current.csv")
    print(f"Total players mapped: {len(final_df)}")
    
    # Show team distribution
    team_counts = final_df['team_name'].value_counts()
    print(f"\n📊 Team Distribution:")
    for team, count in team_counts.head(10).items():
        print(f"  {team}: {count} players")
    
else:
    print("\n❌ Failed to fetch current team data")
    print("Please check internet connection and try again")