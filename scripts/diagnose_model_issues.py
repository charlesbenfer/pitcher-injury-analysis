"""
Diagnose why the survival model has poor C-index performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter

def diagnose_survival_model_issues():
    """Comprehensive diagnosis of survival model problems"""
    
    print("🔍 DIAGNOSING SURVIVAL MODEL ISSUES")
    print("=" * 50)
    
    # Load the corrected dataset
    df = pd.read_csv('data/processed/corrected_survival_dataset_20250902.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
    
    # 1. Check data balance and patterns
    print(f"\n1. DATA BALANCE ANALYSIS:")
    print("-" * 30)
    
    # Time to event distribution by event status
    injured_times = df[df['event'] == 1]['time_to_event']
    censored_times = df[df['event'] == 0]['time_to_event']
    
    print(f"Injured - Mean time: {injured_times.mean():.1f}, Median: {injured_times.median():.1f}")
    print(f"Censored - Mean time: {censored_times.mean():.1f}, Median: {censored_times.median():.1f}")
    
    # This is the key diagnostic - are censored times artificially high?
    if censored_times.mean() > injured_times.mean() * 1.5:
        print("⚠️  WARNING: Censored times much higher than injury times")
        print("   This creates artificial survival patterns")
    
    # 2. Feature relationship analysis
    print(f"\n2. FEATURE-OUTCOME RELATIONSHIPS:")
    print("-" * 35)
    
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    
    # Check if features actually differ between injured/non-injured
    significant_diffs = []
    for col in feature_cols[:6]:  # Check top 6 features
        injured_vals = df[df['event'] == 1][col]
        censored_vals = df[df['event'] == 0][col]
        
        mean_diff = injured_vals.mean() - censored_vals.mean()
        
        # Simple t-test approximation
        from scipy import stats
        try:
            t_stat, p_val = stats.ttest_ind(injured_vals.dropna(), censored_vals.dropna())
            significant = p_val < 0.05
            
            print(f"{col:<20}: Injured={injured_vals.mean():.2f}, Censored={censored_vals.mean():.2f}")
            print(f"{'':20}  Diff={mean_diff:+.2f}, p={p_val:.3f} {'***' if significant else ''}")
            
            if significant:
                significant_diffs.append((col, mean_diff, p_val))
                
        except Exception as e:
            print(f"{col:<20}: Could not compute test")
    
    if len(significant_diffs) == 0:
        print("❌ MAJOR ISSUE: No features significantly differ between groups!")
        print("   This explains why the model can't discriminate")
    else:
        print(f"✓ Found {len(significant_diffs)} features with significant differences")
    
    # 3. Survival curve analysis
    print(f"\n3. SURVIVAL PATTERN ANALYSIS:")
    print("-" * 30)
    
    # Check if we have realistic survival patterns
    kmf = KaplanMeierFitter()
    
    # Overall survival
    kmf.fit(df['time_to_event'], df['event'])
    survival_at_90 = kmf.survival_function_at_times(90).iloc[0]
    survival_at_180 = kmf.survival_function_at_times(180).iloc[0]
    
    print(f"Overall survival at 90 days: {survival_at_90:.2%}")
    print(f"Overall survival at 180 days: {survival_at_180:.2%}")
    
    # Check if survival curve is monotonic decreasing
    surv_func = kmf.survival_function_
    if not surv_func.iloc[:, 0].is_monotonic_decreasing:
        print("⚠️  WARNING: Survival function is not monotonic decreasing")
        print("   This suggests data quality issues")
    
    # 4. Test simple Cox model for comparison
    print(f"\n4. SIMPLE COX MODEL BASELINE:")
    print("-" * 30)
    
    try:
        # Prepare data for Cox model
        cox_data = df[['time_to_event', 'event'] + feature_cols[:5]].dropna()
        
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='time_to_event', event_col='event')
        
        # Compute C-index
        cox_c_index = cph.concordance_index_
        print(f"Cox model C-index: {cox_c_index:.3f}")
        
        # Check which features are significant
        summary = cph.summary
        sig_features = summary[summary['p'] < 0.05]
        print(f"Significant features in Cox model: {len(sig_features)}")
        
        if len(sig_features) > 0:
            print("Top significant features:")
            for idx, row in sig_features.head(3).iterrows():
                direction = "↑ risk" if row['coef'] > 0 else "↓ risk"
                print(f"  {idx}: {row['coef']:+.3f} (p={row['p']:.3f}) {direction}")
        
    except Exception as e:
        print(f"Cox model failed: {e}")
        cox_c_index = np.nan
    
    # 5. Data generation issues
    print(f"\n5. SYNTHETIC DATA QUALITY CHECK:")
    print("-" * 35)
    
    # Check if our synthetic features are too random
    feature_correlations = df[feature_cols].corr()
    avg_abs_corr = np.abs(feature_correlations.values[np.triu_indices_from(feature_correlations.values, k=1)]).mean()
    
    print(f"Average absolute correlation between features: {avg_abs_corr:.3f}")
    if avg_abs_corr < 0.1:
        print("⚠️  WARNING: Features are too uncorrelated (too random)")
        print("   Real pitcher data should have moderate correlations")
    
    # Check if age patterns are realistic
    if 'age_prev' in df.columns:
        age_injury_corr = df['age_prev'].corr(df['event'])
        print(f"Age-injury correlation: {age_injury_corr:+.3f}")
        
        # Age should typically be protective (negative correlation)
        if age_injury_corr > 0:
            print("⚠️  Age positively correlated with injury (unusual)")
        else:
            print("✓ Age negatively correlated with injury (realistic)")
    
    # 6. Recommendations
    print(f"\n📋 DIAGNOSTIC RECOMMENDATIONS:")
    print("-" * 35)
    
    issues_found = []
    
    if censored_times.mean() > injured_times.mean() * 1.5:
        issues_found.append("Artificial censoring pattern")
        print("1. Fix censoring: Use realistic follow-up times")
    
    if len(significant_diffs) == 0:
        issues_found.append("No discriminating features")
        print("2. Feature engineering: Create more realistic performance differences")
        print("   - Use actual performance data instead of synthetic")
        print("   - Add interaction terms")
        print("   - Include injury history")
    
    if cox_c_index < 0.55:
        issues_found.append("Poor baseline performance")
        print("3. Model approach: Try different modeling strategies")
        print("   - Logistic regression for binary outcome")
        print("   - Random Forest for feature interactions")
        print("   - Different survival models")
    
    if avg_abs_corr < 0.1:
        issues_found.append("Unrealistic feature relationships")
        print("4. Data quality: Replace synthetic data with real data")
        print("   - Scrape actual pitcher performance data")
        print("   - Use Baseball Savant, FanGraphs, etc.")
    
    if len(issues_found) == 0:
        print("✓ No obvious issues found - model may need different approach")
    
    print(f"\n🎯 PRIORITY ACTIONS:")
    print("1. Replace synthetic features with real pitcher performance data")
    print("2. Validate injury timing against actual MLB injury reports") 
    print("3. Try simpler models (logistic regression) before complex survival models")
    
    return {
        'injured_mean_time': injured_times.mean(),
        'censored_mean_time': censored_times.mean(),
        'significant_features': len(significant_diffs),
        'cox_c_index': cox_c_index,
        'feature_correlation': avg_abs_corr,
        'issues_found': issues_found
    }

def create_diagnostic_plots():
    """Create diagnostic plots"""
    df = pd.read_csv('data/processed/corrected_survival_dataset_20250902.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Time to event by outcome
    injured = df[df['event'] == 1]['time_to_event']
    censored = df[df['event'] == 0]['time_to_event']
    
    axes[0,0].hist(injured, alpha=0.7, label='Injured', bins=30, color='red')
    axes[0,0].hist(censored, alpha=0.7, label='Censored', bins=30, color='blue')
    axes[0,0].set_title('Time to Event Distribution')
    axes[0,0].legend()
    
    # 2. Survival curve
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(df['time_to_event'], df['event'])
    
    axes[0,1].step(kmf.timeline, kmf.survival_function_.iloc[:, 0], where='post')
    axes[0,1].set_title('Kaplan-Meier Survival Curve')
    axes[0,1].set_xlabel('Days')
    axes[0,1].set_ylabel('Survival Probability')
    
    # 3. Feature boxplots
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    if len(feature_cols) >= 2:
        df_melted = df.melt(
            id_vars=['event'], 
            value_vars=feature_cols[:4], 
            var_name='feature', 
            value_name='value'
        )
        sns.boxplot(data=df_melted, x='feature', y='value', hue='event', ax=axes[1,0])
        axes[1,0].set_title('Feature Distributions by Outcome')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Correlation heatmap
    corr_matrix = df[feature_cols[:6]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlations')
    
    plt.tight_layout()
    plt.savefig('data/processed/model_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Diagnostic plots saved to: data/processed/model_diagnostics.png")

if __name__ == "__main__":
    results = diagnose_survival_model_issues()
    create_diagnostic_plots()
    
    print(f"\nDiagnostic complete. Check the plots and recommendations above.")