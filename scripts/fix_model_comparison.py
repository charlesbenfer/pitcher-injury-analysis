"""
Fix model comparison and compute proper metrics
"""

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

def safe_compute_waic_loo(trace, model_name="model"):
    """Safely compute WAIC/LOO with error handling"""
    try:
        # Try computing WAIC
        waic = az.waic(trace)
        waic_value = waic.elpd_waic
        waic_se = waic.se
        print(f"✓ {model_name} WAIC: {waic_value:.2f} ± {waic_se:.2f}")
    except Exception as e:
        print(f"❌ {model_name} WAIC failed: {str(e)[:60]}...")
        waic_value, waic_se = np.nan, np.nan
    
    try:
        # Try computing LOO
        loo = az.loo(trace)
        loo_value = loo.elpd_loo
        loo_se = loo.se
        print(f"✓ {model_name} LOO: {loo_value:.2f} ± {loo_se:.2f}")
    except Exception as e:
        print(f"❌ {model_name} LOO failed: {str(e)[:60]}...")
        loo_value, loo_se = np.nan, np.nan
    
    return waic_value, waic_se, loo_value, loo_se

def compute_survival_metrics(trace, X_test, y_time_test, y_event_test, model_name="model"):
    """Compute survival-specific metrics"""
    print(f"\nComputing survival metrics for {model_name}...")
    
    try:
        # Extract posterior samples
        if 'beta' in trace.posterior.data_vars:
            beta_samples = trace.posterior['beta'].values.reshape(-1, X_test.shape[1])
            beta_0_samples = trace.posterior['beta_0'].values.reshape(-1)
            
            # Compute risk scores for each posterior sample
            n_samples = min(100, len(beta_samples))  # Use subset for efficiency
            risk_scores = []
            
            for i in range(0, n_samples, max(1, n_samples//20)):
                linear_pred = beta_0_samples[i] + X_test @ beta_samples[i]
                risk_scores.append(linear_pred)
            
            # Average risk scores across samples
            mean_risk_scores = np.mean(risk_scores, axis=0)
            
            # Compute C-index (concordance index)
            try:
                c_index = concordance_index(y_time_test, -mean_risk_scores, y_event_test)
                print(f"✓ {model_name} C-index: {c_index:.3f}")
            except Exception as e:
                print(f"❌ C-index computation failed: {e}")
                c_index = np.nan
            
            # Compute AUC if we have both events and censoring
            if len(np.unique(y_event_test)) > 1:
                try:
                    auc = roc_auc_score(y_event_test, mean_risk_scores)
                    print(f"✓ {model_name} AUC: {auc:.3f}")
                except Exception as e:
                    print(f"❌ AUC computation failed: {e}")
                    auc = np.nan
            else:
                auc = np.nan
            
            return c_index, auc, mean_risk_scores
            
        else:
            print(f"❌ No 'beta' coefficients found in {model_name}")
            return np.nan, np.nan, None
            
    except Exception as e:
        print(f"❌ Survival metrics computation failed: {e}")
        return np.nan, np.nan, None

def analyze_feature_significance(trace, feature_names, model_name="model"):
    """Analyze feature significance with credible intervals"""
    print(f"\nAnalyzing feature significance for {model_name}...")
    
    try:
        if 'beta' in trace.posterior.data_vars:
            # Get posterior summary
            summary = az.summary(trace, var_names=['beta'], hdi_prob=0.95)
            
            # Add feature names
            summary['feature'] = feature_names
            summary['significant'] = (summary['hdi_2.5%'] > 0) | (summary['hdi_97.5%'] < 0)
            
            # Sort by absolute mean coefficient
            summary['abs_mean'] = np.abs(summary['mean'])
            summary_sorted = summary.sort_values('abs_mean', ascending=False)
            
            print(f"Feature significance analysis:")
            print("-" * 60)
            
            significant_features = summary_sorted[summary_sorted['significant']]
            if len(significant_features) > 0:
                print(f"✓ Significant features ({len(significant_features)}):")
                for _, row in significant_features.iterrows():
                    direction = "↑ risk" if row['mean'] > 0 else "↓ risk"
                    print(f"  {row['feature']:<20}: {row['mean']:+.3f} [{row['hdi_2.5%']:+.3f}, {row['hdi_97.5%']:+.3f}] ({direction})")
            else:
                print("❌ No statistically significant features found")
            
            print(f"\nTop features by effect size:")
            for _, row in summary_sorted.head(5).iterrows():
                sig_marker = "***" if row['significant'] else ""
                print(f"  {row['feature']:<20}: {row['mean']:+.3f} ± {row['sd']:.3f} {sig_marker}")
            
            return summary_sorted
            
        else:
            print(f"❌ No coefficients found for significance analysis")
            return None
            
    except Exception as e:
        print(f"❌ Feature significance analysis failed: {e}")
        return None

def comprehensive_model_comparison(traces_dict, X_test, y_time_test, y_event_test, feature_names):
    """Comprehensive model comparison with multiple metrics"""
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for model_name, trace in traces_dict.items():
        print(f"\n📊 Analyzing {model_name.upper()}...")
        print("-" * 40)
        
        # Check convergence
        try:
            rhat = az.rhat(trace)
            max_rhat = float(rhat.max().values)
            print(f"✓ Convergence (max R-hat): {max_rhat:.3f}")
            converged = max_rhat < 1.1
        except:
            max_rhat = np.nan
            converged = False
            print(f"❌ Could not check convergence")
        
        # Compute information criteria
        waic_val, waic_se, loo_val, loo_se = safe_compute_waic_loo(trace, model_name)
        
        # Compute survival metrics
        c_index, auc, risk_scores = compute_survival_metrics(
            trace, X_test, y_time_test, y_event_test, model_name
        )
        
        # Analyze feature significance
        feature_analysis = analyze_feature_significance(trace, feature_names, model_name)
        
        # Store results
        results[model_name] = {
            'max_rhat': max_rhat,
            'converged': converged,
            'waic': waic_val,
            'waic_se': waic_se,
            'loo': loo_val,
            'loo_se': loo_se,
            'c_index': c_index,
            'auc': auc,
            'feature_analysis': feature_analysis,
            'risk_scores': risk_scores
        }
    
    # Summary table
    print(f"\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    
    summary_df = pd.DataFrame({
        model: {
            'Converged': '✓' if res['converged'] else '❌',
            'Max R-hat': f"{res['max_rhat']:.3f}" if not np.isnan(res['max_rhat']) else 'N/A',
            'C-index': f"{res['c_index']:.3f}" if not np.isnan(res['c_index']) else 'N/A',
            'AUC': f"{res['auc']:.3f}" if not np.isnan(res['auc']) else 'N/A',
            'WAIC': f"{res['waic']:.1f}" if not np.isnan(res['waic']) else 'Failed',
            'LOO': f"{res['loo']:.1f}" if not np.isnan(res['loo']) else 'Failed'
        }
        for model, res in results.items()
    }).T
    
    print(summary_df)
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    
    # Find best performing model
    valid_models = {k: v for k, v in results.items() if v['converged'] and not np.isnan(v['c_index'])}
    
    if valid_models:
        best_c_index = max(valid_models.values(), key=lambda x: x['c_index'])['c_index']
        best_models = [k for k, v in valid_models.items() if abs(v['c_index'] - best_c_index) < 0.01]
        
        print(f"🏆 Best performing model(s): {', '.join(best_models)}")
        print(f"   C-index: {best_c_index:.3f}")
        
        # Count significant features
        for model_name in best_models[:2]:  # Top 2 models
            res = results[model_name]
            if res['feature_analysis'] is not None:
                n_sig = res['feature_analysis']['significant'].sum()
                print(f"   {model_name}: {n_sig} significant features")
    else:
        print("❌ No models converged successfully")
    
    return results

# Example usage - you would call this with your actual trace objects
def example_usage():
    """Example of how to use the fixed model comparison"""
    
    # Load your data
    df = pd.read_csv('data/processed/corrected_survival_dataset_20250902.csv')
    
    # Prepare features (same as in your notebook)
    feature_cols = [col for col in df.columns if col.endswith('_prev')]
    # ... your data preparation code ...
    
    # Assume you have traces from your models
    traces_dict = {
        'weibull_aft': None,  # Your Weibull trace
        'cox_ph': None,       # Your Cox trace
        # Add other models...
    }
    
    # Run comprehensive comparison
    # results = comprehensive_model_comparison(traces_dict, X_test, y_time_test, y_event_test, feature_cols)
    
    print("Replace None values with your actual trace objects and run!")

if __name__ == "__main__":
    example_usage()