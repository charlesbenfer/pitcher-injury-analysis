"""
Fixed Weibull survival model for expanded dataset
Handles numerical issues and improves convergence
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    """Load and prepare data with better numerical properties"""
    df = pd.read_csv('data/processed/comprehensive_survival_dataset_20250902.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
    
    # Check for data issues
    print(f"Time to event range: {df['time_to_event'].min()} - {df['time_to_event'].max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Handle extreme values in time_to_event
    df = df[df['time_to_event'] > 0].copy()  # Remove zero times
    
    # Select features and handle missing values
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"Feature columns: {feature_cols}")
    
    # Fill missing values
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Prepare features
    X = df[feature_cols].copy()
    y_time = df['time_to_event'].values
    y_event = df['event'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    print(f"Final dataset: {len(X_scaled)} observations, {len(feature_cols)} features")
    print(f"Events: {y_event.sum()}, Censored: {(1-y_event).sum()}")
    
    return X_scaled, y_time, y_event, feature_cols

def create_robust_weibull_model(X, y_time, y_event):
    """Create a more robust Weibull model"""
    
    print("Creating robust Weibull AFT model...")
    
    with pm.Model() as weibull_model:
        # Data containers
        X_data = pm.Data('X', X.values)
        y_time_data = pm.Data('y_time', y_time)
        y_event_data = pm.Data('y_event', y_event)
        
        # More conservative priors to avoid numerical issues
        beta_0 = pm.Normal('beta_0', mu=np.log(np.median(y_time)), sigma=0.5)
        beta = pm.Normal('beta', mu=0, sigma=0.25, shape=X.shape[1])
        
        # Shape parameter with better constraint
        alpha = pm.Gamma('alpha', alpha=2, beta=2)  # More concentrated around 1
        
        # Linear predictor
        mu = beta_0 + pm.math.dot(X_data, beta)
        
        # Scale parameter (lambda in Weibull parameterization)
        lambda_param = pm.math.exp(mu)
        
        # Separate observed and censored cases more carefully
        n_obs = len(y_time)
        
        # Create likelihood components
        log_likelihood = pm.math.zeros(n_obs)
        
        # For observed events (event = 1)
        obs_mask = y_event_data == 1
        log_pdf = (pm.math.log(alpha) - pm.math.log(lambda_param) + 
                   (alpha - 1) * (pm.math.log(y_time_data) - pm.math.log(lambda_param)) - 
                   (y_time_data / lambda_param) ** alpha)
        
        # For censored events (event = 0) 
        cens_mask = y_event_data == 0
        log_sf = -(y_time_data / lambda_param) ** alpha  # Log survival function
        
        # Combine likelihoods
        total_ll = pm.math.sum(obs_mask * log_pdf + cens_mask * log_sf)
        
        # Add as potential
        pm.Potential('weibull_likelihood', total_ll)
        
        print(f"Model created with {X.shape[1]} features")
        
    return weibull_model

def sample_model_robust(model, draws=1000, tune=500):
    """Sample with robust settings"""
    
    print("Starting robust sampling...")
    
    with model:
        # Use more conservative sampling settings
        try:
            # First try with default settings
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=2,  # Fewer chains initially
                cores=2,
                target_accept=0.8,  # Less aggressive
                max_treedepth=8,
                init='auto',
                return_inferencedata=True
            )
            print("Sampling completed successfully!")
            return trace
            
        except Exception as e:
            print(f"Standard sampling failed: {e}")
            print("Trying with MAP initialization...")
            
            # Try with MAP initialization
            map_estimate = pm.find_MAP()
            print(f"MAP estimate found: {map_estimate}")
            
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=2,
                cores=1,  # Single core for stability
                target_accept=0.75,
                init=map_estimate,
                return_inferencedata=True
            )
            print("Sampling with MAP initialization completed!")
            return trace

def analyze_results(trace, feature_names):
    """Analyze sampling results"""
    
    print("\n" + "="*60)
    print("SAMPLING RESULTS SUMMARY")
    print("="*60)
    
    # Check convergence
    print(f"R-hat summary:")
    rhat = az.rhat(trace)
    print(f"  Max R-hat: {rhat.max().values:.3f}")
    print(f"  Mean R-hat: {rhat.mean().values:.3f}")
    
    if rhat.max().values > 1.1:
        print("  ⚠️  WARNING: Some parameters may not have converged (R-hat > 1.1)")
    else:
        print("  ✓ All parameters converged (R-hat < 1.1)")
    
    # Effective sample size
    ess = az.ess(trace)
    print(f"\nEffective sample size:")
    print(f"  Min ESS: {ess.min().values:.0f}")
    print(f"  Mean ESS: {ess.mean().values:.0f}")
    
    # Parameter summaries
    print(f"\nParameter estimates:")
    summary = az.summary(trace, round_to=3)
    print(summary)
    
    # Feature importance (based on posterior means)
    if 'beta' in trace.posterior.data_vars:
        beta_means = trace.posterior['beta'].mean(['chain', 'draw']).values
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': beta_means,
            'abs_coefficient': np.abs(beta_means)
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\nFeature importance (by |coefficient|):")
        print(feature_importance.head(10))
    
    return summary, feature_importance if 'beta' in trace.posterior.data_vars else None

def main():
    """Main execution"""
    print("Loading and preparing data...")
    X, y_time, y_event, feature_names = load_and_prepare_data()
    
    print("Creating robust Weibull model...")
    model = create_robust_weibull_model(X, y_time, y_event)
    
    print("Sampling from posterior...")
    trace = sample_model_robust(model, draws=1000, tune=500)
    
    print("Analyzing results...")
    summary, importance = analyze_results(trace, feature_names)
    
    # Save results
    trace.to_netcdf('data/processed/weibull_trace_robust.nc')
    summary.to_csv('data/processed/weibull_summary_robust.csv')
    if importance is not None:
        importance.to_csv('data/processed/feature_importance_robust.csv', index=False)
    
    print("\nResults saved:")
    print("  - data/processed/weibull_trace_robust.nc")
    print("  - data/processed/weibull_summary_robust.csv")
    print("  - data/processed/feature_importance_robust.csv")
    
    return trace, summary, importance

if __name__ == "__main__":
    trace, summary, importance = main()