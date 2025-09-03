"""
Fix the risk scoring calibration in the pitcher dashboard
Steps:
1. Calculate proper risk quartiles from training data
2. Validate risk scoring logic against actual outcomes  
3. Ensure realistic risk stratification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def calculate_proper_risk_quartiles():
    """
    Step 1: Calculate proper risk quartiles from training data
    """
    print("🔧 STEP 1: CALCULATING PROPER RISK QUARTILES")
    print("=" * 45)
    
    # Load data
    df = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
    
    # Use the same features as dashboard
    dashboard_features = [
        'age_prev', 'w_prev', 'l_prev', 'era_prev', 'g_prev', 'gs_prev', 'ip_prev',
        'h_prev', 'r_prev', 'er_prev', 'hr_prev', 'bb_prev', 'so_prev', 'whip_prev',
        'k_per_9_prev', 'bb_per_9_prev', 'hr_per_9_prev', 'fip_prev', 'war_prev',
        'high_workload_prev', 'veteran_prev', 'high_era_prev'
    ]
    
    available_features = [f for f in dashboard_features if f in df.columns]
    print(f"Available features: {len(available_features)}")
    
    # Prepare data
    X = df[available_features].fillna(0)
    y_time = df['time_to_event'].values
    y_event = df['event'].values
    
    # Split data (same as original analysis)
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=42, stratify=y_event
    )
    
    print(f"Training set: {len(X_train)} observations, {y_event_train.sum()} events")
    
    # Use the same simplified scoring as dashboard
    feature_weights = {
        'age_prev': -0.03,      # Older = higher risk
        'g_prev': 0.02,         # More games = protective (durability)  
        'veteran_prev': 0.01,   # Veterans = slightly protective
        'era_prev': -0.01,      # Higher ERA = higher risk
        'ip_prev': 0.001,       # More IP = protective (durability)
        'war_prev': 0.005,      # Higher WAR = protective
        'high_workload_prev': -0.005  # High workload = risk
    }
    
    beta_0 = 5.0
    
    # Calculate risk scores for training data
    def calculate_risk_score(pitcher_stats):
        linear_pred = beta_0
        for feature, weight in feature_weights.items():
            if feature in pitcher_stats:
                linear_pred += weight * pitcher_stats[feature]
        return -linear_pred  # AFT: higher linear pred = lower risk
    
    training_risk_scores = []
    for idx, row in X_train.iterrows():
        pitcher_stats = row.to_dict()
        risk_score = calculate_risk_score(pitcher_stats)
        training_risk_scores.append(risk_score)
    
    training_risk_scores = np.array(training_risk_scores)
    
    # Calculate quartiles from training data
    quartiles = np.percentile(training_risk_scores, [25, 50, 75])
    
    print(f"Training risk score statistics:")
    print(f"  Min: {training_risk_scores.min():.3f}")
    print(f"  Q1:  {quartiles[0]:.3f}")
    print(f"  Q2:  {quartiles[1]:.3f}")
    print(f"  Q3:  {quartiles[2]:.3f}")
    print(f"  Max: {training_risk_scores.max():.3f}")
    
    # Use quartiles as risk thresholds
    risk_quartiles = [quartiles[0], quartiles[1], quartiles[2]]
    
    print(f"\n📊 PROPER RISK QUARTILES:")
    print(f"  Low Risk:      ≤ {risk_quartiles[0]:.3f}")
    print(f"  Moderate Risk: ≤ {risk_quartiles[1]:.3f}")
    print(f"  High Risk:     ≤ {risk_quartiles[2]:.3f}")
    print(f"  Very High Risk: > {risk_quartiles[2]:.3f}")
    
    return risk_quartiles, training_risk_scores, X_train, y_event_train

def validate_risk_scoring(risk_quartiles, training_risk_scores, y_event_train):
    """
    Step 2: Validate risk scoring logic against actual outcomes
    """
    print(f"\n🎯 STEP 2: VALIDATING RISK SCORING LOGIC")
    print("=" * 40)
    
    # Categorize training data by risk level
    def get_risk_category(score, quartiles):
        if score <= quartiles[0]:
            return "Low"
        elif score <= quartiles[1]:
            return "Moderate"
        elif score <= quartiles[2]:
            return "High"
        else:
            return "Very High"
    
    risk_categories = [get_risk_category(score, risk_quartiles) for score in training_risk_scores]
    
    # Create validation dataframe
    validation_df = pd.DataFrame({
        'risk_score': training_risk_scores,
        'risk_category': risk_categories,
        'actual_injury': y_event_train
    })
    
    # Calculate injury rates by risk category
    risk_analysis = validation_df.groupby('risk_category')['actual_injury'].agg(['count', 'sum', 'mean']).round(3)
    risk_analysis['injury_rate'] = risk_analysis['mean']
    
    print("Risk Category Validation:")
    print("-" * 25)
    for category in ['Low', 'Moderate', 'High', 'Very High']:
        if category in risk_analysis.index:
            stats = risk_analysis.loc[category]
            print(f"{category:<12}: {stats['count']:3.0f} pitchers, {stats['sum']:3.0f} injuries ({stats['injury_rate']:5.1%})")
        else:
            print(f"{category:<12}: 0 pitchers")
    
    # Validate that higher risk = higher injury rates
    print(f"\n📈 RISK GRADIENT VALIDATION:")
    injury_rates = []
    for category in ['Low', 'Moderate', 'High', 'Very High']:
        if category in risk_analysis.index:
            rate = risk_analysis.loc[category, 'injury_rate']
            injury_rates.append(rate)
            print(f"  {category}: {rate:.1%}")
        else:
            injury_rates.append(0)
    
    # Check if injury rates increase with risk level
    is_monotonic = all(injury_rates[i] <= injury_rates[i+1] for i in range(len(injury_rates)-1))
    
    if is_monotonic:
        print("✅ Risk scoring validated: Higher risk categories have higher injury rates")
    else:
        print("⚠️  Risk scoring needs adjustment: Injury rates not monotonic")
    
    return validation_df, risk_analysis, is_monotonic

def ensure_realistic_stratification(risk_quartiles):
    """
    Step 3: Ensure realistic risk stratification on full dataset
    """
    print(f"\n📊 STEP 3: TESTING REALISTIC RISK STRATIFICATION")
    print("=" * 48)
    
    # Load full dataset
    df = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
    
    # Filter to 2023 data (same as dashboard)
    roster_2023 = df[(df['season'] == 2023) & (df['g_prev'] >= 10)].copy()
    
    print(f"Testing on 2023 roster: {len(roster_2023)} pitchers")
    
    available_features = [
        'age_prev', 'w_prev', 'l_prev', 'era_prev', 'g_prev', 'gs_prev', 'ip_prev',
        'h_prev', 'r_prev', 'er_prev', 'hr_prev', 'bb_prev', 'so_prev', 'whip_prev',
        'k_per_9_prev', 'bb_per_9_prev', 'hr_per_9_prev', 'fip_prev', 'war_prev',
        'high_workload_prev', 'veteran_prev', 'high_era_prev'
    ]
    available_features = [f for f in available_features if f in df.columns]
    
    # Calculate risk scores for 2023 roster
    feature_weights = {
        'age_prev': -0.03,
        'g_prev': 0.02,
        'veteran_prev': 0.01,
        'era_prev': -0.01,
        'ip_prev': 0.001,
        'war_prev': 0.005,
        'high_workload_prev': -0.005
    }
    
    beta_0 = 5.0
    
    def calculate_risk_score(pitcher_stats):
        linear_pred = beta_0
        for feature, weight in feature_weights.items():
            if feature in pitcher_stats:
                linear_pred += weight * pitcher_stats[feature]
        return -linear_pred
    
    def get_risk_category(score, quartiles):
        if score <= quartiles[0]:
            return "Low", 0
        elif score <= quartiles[1]:
            return "Moderate", 1
        elif score <= quartiles[2]:
            return "High", 2
        else:
            return "Very High", 3
    
    # Apply to 2023 roster
    roster_results = []
    for idx, pitcher in roster_2023.iterrows():
        pitcher_stats = pitcher[available_features].to_dict()
        risk_score = calculate_risk_score(pitcher_stats)
        risk_category, alert_level = get_risk_category(risk_score, risk_quartiles)
        
        roster_results.append({
            'player_name': pitcher['player_name'],
            'risk_score': risk_score,
            'risk_category': risk_category,
            'alert_level': alert_level,
            'actual_injury': pitcher['event']
        })
    
    results_df = pd.DataFrame(roster_results)
    
    # Check distribution
    distribution = results_df['risk_category'].value_counts()
    print(f"\n2023 Roster Risk Distribution:")
    print("-" * 30)
    for category in ['Very High', 'High', 'Moderate', 'Low']:
        count = distribution.get(category, 0)
        pct = count / len(results_df) * 100
        color_map = {'Very High': '🔴', 'High': '🟠', 'Moderate': '🟡', 'Low': '🟢'}
        print(f"  {color_map[category]} {category:<12}: {count:2d} pitchers ({pct:4.1f}%)")
    
    # Check if distribution is reasonable (not 100% low risk)
    high_risk_count = len(results_df[results_df['alert_level'] >= 2])
    total_injuries = results_df['actual_injury'].sum()
    
    print(f"\n📊 DISTRIBUTION VALIDATION:")
    print(f"  High-risk pitchers (alert level 2+): {high_risk_count}/{len(results_df)} ({high_risk_count/len(results_df)*100:.1f}%)")
    print(f"  Actual injuries in 2023: {total_injuries}")
    
    if high_risk_count == 0:
        print("❌ ISSUE: No high-risk pitchers identified - thresholds too restrictive")
        return False, risk_quartiles, results_df
    elif high_risk_count > len(results_df) * 0.5:
        print("⚠️  WARNING: Too many high-risk pitchers - thresholds too lenient") 
        return False, risk_quartiles, results_df
    else:
        print("✅ REALISTIC: Balanced risk distribution achieved")
        return True, risk_quartiles, results_df

def generate_corrected_risk_quartiles():
    """
    Generate the corrected risk quartiles for the dashboard
    """
    print(f"\n🔧 GENERATING CORRECTED RISK SCORER")
    print("=" * 38)
    
    # Calculate proper quartiles
    risk_quartiles, training_scores, X_train, y_event_train = calculate_proper_risk_quartiles()
    
    # Validate logic
    validation_df, risk_analysis, is_monotonic = validate_risk_scoring(risk_quartiles, training_scores, y_event_train)
    
    # Test stratification
    is_realistic, final_quartiles, test_results = ensure_realistic_stratification(risk_quartiles)
    
    if is_monotonic and is_realistic:
        print(f"\n✅ RISK SCORING CALIBRATION COMPLETE")
        print(f"   Proper quartiles: {final_quartiles}")
        print(f"   Monotonic risk gradient: {is_monotonic}")
        print(f"   Realistic distribution: {is_realistic}")
        
        # Generate corrected dashboard code
        corrected_code = f'''
# CORRECTED RISK QUARTILES (replace in dashboard)
risk_quartiles = [{final_quartiles[0]:.3f}, {final_quartiles[1]:.3f}, {final_quartiles[2]:.3f}]

# Initialize corrected risk scorer
risk_scorer = PitcherRiskScorer(dashboard_features, risk_quartiles=risk_quartiles)
'''
        
        print(f"\n📋 CORRECTED DASHBOARD CODE:")
        print(corrected_code)
        
        return final_quartiles, True
        
    else:
        print(f"\n❌ CALIBRATION FAILED")
        print(f"   Monotonic: {is_monotonic}")
        print(f"   Realistic: {is_realistic}")
        print("   Manual adjustment of feature weights may be needed")
        
        return risk_quartiles, False

if __name__ == "__main__":
    print("🏥 PITCHER RISK SCORING CALIBRATION")
    print("=" * 40)
    
    final_quartiles, success = generate_corrected_risk_quartiles()
    
    if success:
        print(f"\n🎯 CALIBRATION SUCCESSFUL!")
        print(f"Use these quartiles in your dashboard:")
        print(f"risk_quartiles = {final_quartiles}")
    else:
        print(f"\n⚠️  CALIBRATION NEEDS REFINEMENT")
        print("Review feature weights and thresholds")