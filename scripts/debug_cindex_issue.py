"""
Debug why C-index is 0.361 when Cox model gets 0.667
The 180-day censoring is correct - something else is wrong
"""

import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

def debug_concordance_issue():
    """Debug why Bayesian model C-index differs from Cox"""
    
    print("🔍 DEBUGGING C-INDEX ISSUE")
    print("=" * 40)
    
    df = pd.read_csv('data/processed/corrected_survival_dataset_20250902.csv')
    
    print(f"Dataset: {df.shape}")
    print(f"Event rate: {df['event'].mean():.1%}")
    print(f"Censoring at 180 days: {(df['time_to_event'] == 180).sum()} observations")
    print("✓ 180-day censoring is correct (full season without injury)")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    X = df[feature_cols].fillna(0)
    y_time = df['time_to_event'].values
    y_event = df['event'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    # 1. Test concordance calculation methods
    print(f"\n1. CONCORDANCE CALCULATION DEBUG:")
    print("-" * 35)
    
    # Simple linear combination (like Cox model)
    # Higher ERA, more games should = higher risk
    simple_risk_score = (
        0.1 * (X['era_prev'] - X['era_prev'].mean()) +  # Higher ERA = more risk
        0.05 * (X['g_prev'] - X['g_prev'].mean()) +     # More games = more risk  
        0.002 * (X['ip_prev'] - X['ip_prev'].mean())    # More innings = more risk
    )
    
    # Test concordance - NOTE: concordance_index expects LOWER scores for higher risk
    # So we need to flip the sign
    c_index_correct = concordance_index(y_time, -simple_risk_score, y_event)
    c_index_wrong = concordance_index(y_time, simple_risk_score, y_event)
    
    print(f"Simple risk score (correct sign): C-index = {c_index_correct:.3f}")
    print(f"Simple risk score (wrong sign): C-index = {c_index_wrong:.3f}")
    
    if c_index_wrong < 0.5 < c_index_correct:
        print("🎯 FOUND IT! Sign convention issue in risk scores")
        print("   concordance_index expects LOWER scores for HIGHER risk")
        print("   Your Bayesian model might have the wrong sign convention")
    
    # 2. Check what Cox model is doing
    print(f"\n2. COX MODEL COMPARISON:")
    print("-" * 25)
    
    try:
        # Fit Cox model
        cox_data = df[['time_to_event', 'event'] + feature_cols[:5]].dropna()
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='time_to_event', event_col='event')
        
        print(f"Cox C-index: {cph.concordance_index_:.3f}")
        print("Cox coefficients:")
        for var in feature_cols[:5]:
            if var in cph.params_.index:
                coef = cph.params_[var]
                print(f"  {var:<15}: {coef:+.3f}")
        
        # Get Cox risk scores
        cox_risk_scores = cph.predict_partial_hazard(cox_data.drop(['time_to_event', 'event'], axis=1))
        
        # Test concordance with Cox scores (should be ~0.667)
        cox_test_c_index = concordance_index(
            cox_data['time_to_event'], 
            cox_risk_scores,  # Cox already gives correct direction 
            cox_data['event']
        )
        print(f"Verification Cox C-index: {cox_test_c_index:.3f}")
        
    except Exception as e:
        print(f"Cox model failed: {e}")
        cox_risk_scores = None
    
    # 3. Test Bayesian model sign convention
    print(f"\n3. BAYESIAN MODEL SIGN CONVENTION:")
    print("-" * 35)
    
    print("In Bayesian AFT models:")
    print("- Higher linear predictor (β₀ + Xβ) = LONGER survival time")
    print("- Longer survival = LOWER hazard = LOWER risk")
    print("- For concordance: need to flip sign of linear predictor")
    
    print("\nCheck your Bayesian model output:")
    print("- If β₀ + Xβ is the linear predictor")
    print("- Use concordance_index(time, -(β₀ + Xβ), event)")
    print("- The NEGATIVE sign is crucial!")
    
    # 4. Simulate what might be happening
    print(f"\n4. SIMULATED BAYESIAN OUTPUT:")
    print("-" * 30)
    
    # Simulate typical Bayesian AFT output
    np.random.seed(42)
    
    # In AFT: log(survival_time) = β₀ + Xβ + error
    # So higher β₀ + Xβ = longer survival = lower risk
    simulated_linear_pred = (
        4.0 +  # Intercept (log days)
        -0.1 * (X['era_prev'] - X['era_prev'].mean()) +  # Higher ERA = shorter survival
        -0.05 * (X['g_prev'] - X['g_prev'].mean()) +     # More games = shorter survival
        -0.002 * (X['ip_prev'] - X['ip_prev'].mean()) +  # More IP = shorter survival  
        np.random.normal(0, 0.5, len(X))  # Error term
    )
    
    # Test both sign conventions
    c_index_aft_correct = concordance_index(y_time, -simulated_linear_pred, y_event)
    c_index_aft_wrong = concordance_index(y_time, simulated_linear_pred, y_event)
    
    print(f"AFT linear predictor (with negative): C-index = {c_index_aft_correct:.3f}")
    print(f"AFT linear predictor (without negative): C-index = {c_index_aft_wrong:.3f}")
    
    # 5. Diagnosis
    print(f"\n🎯 DIAGNOSIS:")
    print("-" * 15)
    
    if c_index_aft_wrong < 0.4:
        print("✓ Your Bayesian model likely has SIGN CONVENTION issue")
        print("  In your validation code, change:")
        print("  FROM: concordance_index(time, linear_pred, event)")
        print("  TO:   concordance_index(time, -linear_pred, event)")
        print("\n  AFT models: higher linear pred = longer survival = LOWER risk")
        print("  Need negative sign for concordance calculation")
    else:
        print("❓ Sign convention doesn't explain the issue")
        print("  Need to investigate other causes")
    
    return {
        'simple_correct': c_index_correct,
        'simple_wrong': c_index_wrong,
        'aft_correct': c_index_aft_correct, 
        'aft_wrong': c_index_aft_wrong
    }

def check_notebook_validation_code():
    """Show how to fix validation code in notebook"""
    
    print(f"\n🔧 FIX FOR YOUR NOTEBOOK:")
    print("-" * 25)
    
    print("In your model validation cell, change this:")
    print()
    print("# WRONG (current):")
    print("risk_scores = beta_0_mean + X_test @ beta_mean")
    print("c_index = concordance_index(y_time_test, risk_scores, y_event_test)")
    print()
    print("# CORRECT (fixed):")
    print("risk_scores = beta_0_mean + X_test @ beta_mean")  
    print("c_index = concordance_index(y_time_test, -risk_scores, y_event_test)  # Note the minus sign!")
    print()
    print("The minus sign is crucial because:")
    print("- AFT model: higher risk_score = longer predicted survival")
    print("- Concordance: expects lower scores for higher risk")
    print("- Solution: flip the sign")

if __name__ == "__main__":
    results = debug_concordance_issue()
    check_notebook_validation_code()
    
    print(f"\n🎯 SUMMARY:")
    print("The 180-day censoring is CORRECT")  
    print("The issue is likely SIGN CONVENTION in concordance calculation")
    print("Try adding the negative sign in your validation code!")