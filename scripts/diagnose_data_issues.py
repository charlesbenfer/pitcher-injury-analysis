"""
Diagnose data issues causing Weibull model failures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_data_issues():
    """Diagnose potential issues with the survival dataset"""
    
    print("Loading comprehensive dataset...")
    df = pd.read_csv('data/processed/comprehensive_survival_dataset_20250902.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic stats
    print(f"\nBasic statistics:")
    print(f"Event rate: {df['event'].mean():.2%}")
    print(f"Events: {df['event'].sum()}, Censored: {(df['event'] == 0).sum()}")
    
    # Time to event issues
    print(f"\nTime to event analysis:")
    print(f"Range: {df['time_to_event'].min()} to {df['time_to_event'].max()}")
    print(f"Mean: {df['time_to_event'].mean():.1f}")
    print(f"Median: {df['time_to_event'].median():.1f}")
    print(f"Zero times: {(df['time_to_event'] == 0).sum()}")
    print(f"Negative times: {(df['time_to_event'] < 0).sum()}")
    
    # Check for extreme values
    print(f"\nExtreme values check:")
    time_q99 = df['time_to_event'].quantile(0.99)
    time_q01 = df['time_to_event'].quantile(0.01)
    print(f"1st percentile: {time_q01:.1f}")
    print(f"99th percentile: {time_q99:.1f}")
    
    # Feature analysis
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # Check for missing values
    print(f"\nMissing values:")
    missing = df[feature_cols].isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values in features")
    
    # Check for infinite values
    print(f"\nInfinite values:")
    for col in feature_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"{col}: {inf_count} infinite values")
    
    # Feature distributions
    print(f"\nFeature distribution summary:")
    feature_stats = df[feature_cols].describe()
    print(feature_stats.loc[['min', 'max', 'mean', 'std']])
    
    # Identify potential issues
    issues = []
    
    # Check for zero/negative times
    if (df['time_to_event'] <= 0).any():
        issues.append(f"Found {(df['time_to_event'] <= 0).sum()} zero/negative time values")
    
    # Check event rate
    event_rate = df['event'].mean()
    if event_rate > 0.95:
        issues.append(f"Very high event rate ({event_rate:.1%}) - may need more censored observations")
    
    # Check for extreme feature values
    for col in feature_cols:
        if df[col].std() == 0:
            issues.append(f"Feature {col} has zero variance")
        elif df[col].std() > 1000:
            issues.append(f"Feature {col} has very large variance ({df[col].std():.1f})")
    
    # Print issues
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print(f"\n✓ No obvious data issues found")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    
    if (df['time_to_event'] <= 0).any():
        print("1. Remove observations with time_to_event <= 0")
    
    if event_rate > 0.9:
        print("2. Consider adding more censored observations or using a different model")
        print("   - Current model assumes too many injuries")
        print("   - Real survival data typically has 20-50% event rates")
    
    print("3. Standardize all features before modeling")
    print("4. Use more conservative priors in Weibull model")
    print("5. Consider log-transforming time_to_event if very skewed")
    
    # Save diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time to event distribution
    axes[0,0].hist(df['time_to_event'], bins=50, alpha=0.7)
    axes[0,0].set_title('Time to Event Distribution')
    axes[0,0].set_xlabel('Days')
    
    # Event rate by time
    time_bins = pd.cut(df['time_to_event'], bins=20)
    event_rate_by_time = df.groupby(time_bins)['event'].mean()
    axes[0,1].bar(range(len(event_rate_by_time)), event_rate_by_time.values)
    axes[0,1].set_title('Event Rate by Time Bins')
    axes[0,1].set_ylabel('Event Rate')
    
    # Feature correlation
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols[:8]].corr()  # Limit to first 8 features
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Feature Correlations')
    
    # Event distribution by season
    if 'season' in df.columns:
        season_events = df.groupby('season')['event'].agg(['count', 'sum', 'mean'])
        axes[1,1].bar(season_events.index, season_events['mean'])
        axes[1,1].set_title('Event Rate by Season')
        axes[1,1].set_xlabel('Season')
        axes[1,1].set_ylabel('Event Rate')
    
    plt.tight_layout()
    plt.savefig('data/processed/data_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plots saved to: data/processed/data_diagnostics.png")
    
    # Create cleaned dataset
    print(f"\n🔧 CREATING CLEANED DATASET:")
    df_clean = df.copy()
    
    # Remove problematic observations
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['time_to_event'] > 0]
    print(f"Removed {initial_count - len(df_clean)} observations with time_to_event <= 0")
    
    # Handle missing values
    for col in feature_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median ({median_val:.2f})")
    
    # Cap extreme values (optional)
    for col in feature_cols:
        if df_clean[col].std() > 0:
            q99 = df_clean[col].quantile(0.99)
            q01 = df_clean[col].quantile(0.01)
            n_capped = ((df_clean[col] > q99) | (df_clean[col] < q01)).sum()
            if n_capped > 0:
                df_clean[col] = np.clip(df_clean[col], q01, q99)
                print(f"Capped {n_capped} extreme values in {col}")
    
    # Save cleaned dataset
    output_file = 'data/processed/survival_dataset_cleaned.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file}")
    print(f"Final shape: {df_clean.shape}")
    print(f"Final event rate: {df_clean['event'].mean():.2%}")
    
    return df_clean

if __name__ == "__main__":
    df_clean = diagnose_data_issues()