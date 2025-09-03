#!/usr/bin/env python
# coding: utf-8

# # 2024 Holdout Validation and 2025 Projections
# 
# External validation on unseen 2024 data and forward projections for 2025 season.

# In[ ]:


import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load full dataset
df = pd.read_csv('../data/processed/survival_dataset_lagged_enhanced.csv')
print(f"Loaded {len(df)} pitcher-seasons from 2019-2024")
print(f"Season distribution:\n{df['season'].value_counts().sort_index()}")


# ## 1. Split Data: 2019-2023 Training vs 2024 Holdout

# In[ ]:


# Split data chronologically
train_data = df[df['season'] < 2024].copy()
holdout_2024 = df[df['season'] == 2024].copy()

print(f"Training data (2019-2023): {len(train_data)} observations")
print(f"Holdout data (2024): {len(holdout_2024)} observations")
print(f"Training injury rate: {train_data['event'].mean():.1%}")
print(f"2024 injury rate: {holdout_2024['event'].mean():.1%}")


# ## 2. Retrain Model on 2019-2023 Data

# In[ ]:


# Features for modeling (lagged variables only)
features = [
    'age_prev', 'g_prev', 'veteran_prev', 'era_prev', 
    'ip_prev', 'war_prev', 'high_workload_prev'
]

# Prepare training data
X_train = train_data[features].copy()
y_time_train = train_data['time_to_event'].values
y_event_train = train_data['event'].values

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Training on {len(X_train)} observations with {len(features)} features")
print(f"Features: {features}")


# In[ ]:


# Retrain Bayesian Weibull AFT model on 2019-2023 data
with pm.Model() as retrain_model:
    # Horseshoe priors for feature selection
    tau = pm.HalfNormal('tau', sigma=1)
    lambda_coef = pm.HalfNormal('lambda', sigma=1, shape=X_train_scaled.shape[1])
    
    # Coefficients with horseshoe prior
    beta = pm.Normal('beta', mu=0, sigma=tau * lambda_coef, shape=X_train_scaled.shape[1])
    beta_0 = pm.Normal('beta_0', mu=5.0, sigma=1)
    
    # Weibull shape parameter
    alpha = pm.Gamma('alpha', alpha=2, beta=1)
    
    # Linear predictor (log-scale parameter for Weibull)
    mu = beta_0 + pm.math.dot(X_train_scaled, beta)
    
    # Weibull likelihood
    obs = pm.Weibull('obs', alpha=alpha, beta=pm.math.exp(mu), 
                     observed=y_time_train, censored=1-y_event_train)
    
    # Fit model
    trace_retrain = pm.sample(2000, tune=1000, random_seed=42, 
                              target_accept=0.95, return_inferencedata=True)

print("✅ Model retrained on 2019-2023 data")


# ## 3. Validate on 2024 Holdout Data

# In[ ]:


# Prepare holdout data
X_holdout = holdout_2024[features].copy()
X_holdout_scaled = scaler.transform(X_holdout)
y_time_holdout = holdout_2024['time_to_event'].values
y_event_holdout = holdout_2024['event'].values

# Generate predictions for 2024
posterior = trace_retrain.posterior
beta_samples = posterior['beta'].values.reshape(-1, len(features))
beta_0_samples = posterior['beta_0'].values.flatten()
alpha_samples = posterior['alpha'].values.flatten()

# Predict survival times (median of posterior predictive)
linear_pred = beta_0_samples.mean() + X_holdout_scaled @ beta_samples.mean(axis=0)
predicted_times = np.exp(linear_pred)

# Calculate risk scores (negative log survival time for concordance)
risk_scores_2024 = -np.log(predicted_times)

print(f"Generated predictions for {len(holdout_2024)} pitchers in 2024")


# In[ ]:


# Calculate external validation metrics
c_index_2024 = concordance_index(y_time_holdout, -risk_scores_2024, y_event_holdout)

print(f"🎯 2024 Holdout Validation Results:")
print(f"C-index: {c_index_2024:.3f}")
print(f"Injuries in 2024: {y_event_holdout.sum()}/{len(y_event_holdout)} ({y_event_holdout.mean():.1%})")

# Risk calibration check
holdout_2024_results = holdout_2024.copy()
holdout_2024_results['risk_score'] = risk_scores_2024
holdout_2024_results['predicted_time'] = predicted_times

# Create risk quartiles
risk_quartiles = np.percentile(risk_scores_2024, [25, 50, 75])
holdout_2024_results['risk_quartile'] = pd.cut(
    risk_scores_2024, 
    bins=[-np.inf] + list(risk_quartiles) + [np.inf],
    labels=['Low', 'Moderate', 'High', 'Very High']
)

# Validation by risk quartile
validation_results = holdout_2024_results.groupby('risk_quartile').agg({
    'event': ['count', 'sum', 'mean'],
    'time_to_event': 'mean'
}).round(3)

validation_results.columns = ['Count', 'Injuries', 'Injury_Rate', 'Avg_Time']
print(f"\n📊 Risk Calibration on 2024 Data:")
print(validation_results)


# ## 4. Create 2025 Season Projections

# In[ ]:


# Create 2025 projections using 2024 stats as lagged features
projections_2025 = holdout_2024.copy()

# Create 2025 projection features (2024 becomes "prev" for 2025)
projection_features = {
    'age_prev': projections_2025['age'] + 1,  # Age up by 1 year
    'g_prev': projections_2025['g'],
    'veteran_prev': (projections_2025['age'] + 1 >= 30).astype(int),
    'era_prev': projections_2025['era'],
    'ip_prev': projections_2025['ip'],
    'war_prev': projections_2025['war'],
    'high_workload_prev': projections_2025['high_workload']
}

# Create projection dataframe
X_projection = pd.DataFrame(projection_features)
X_projection_scaled = scaler.transform(X_projection)

# Generate 2025 risk predictions
linear_pred_2025 = beta_0_samples.mean() + X_projection_scaled @ beta_samples.mean(axis=0)
predicted_times_2025 = np.exp(linear_pred_2025)
risk_scores_2025 = -np.log(predicted_times_2025)

print(f"Generated 2025 projections for {len(projections_2025)} active pitchers")


# In[ ]:


# Create comprehensive 2025 projections dataset
projections_2025['season'] = 2025
projections_2025['risk_score_2025'] = risk_scores_2025
projections_2025['predicted_time_2025'] = predicted_times_2025

# Add 2025 projection features
for feature, values in projection_features.items():
    projections_2025[feature] = values

# Risk categorization for 2025
# Use training data quartiles for consistency
train_risk_scores = -(beta_0_samples.mean() + X_train_scaled @ beta_samples.mean(axis=0))
train_risk_quartiles = np.percentile(train_risk_scores, [25, 50, 75])

projections_2025['risk_category_2025'] = pd.cut(
    risk_scores_2025,
    bins=[-np.inf] + list(train_risk_quartiles) + [np.inf],
    labels=['Low', 'Moderate', 'High', 'Very High']
)

print(f"\n⚾ 2025 Season Risk Distribution:")
print(projections_2025['risk_category_2025'].value_counts())

# Save 2025 projections
projections_2025.to_csv('../data/processed/pitcher_projections_2025.csv', index=False)
print(f"\n💾 Saved 2025 projections to data/processed/pitcher_projections_2025.csv")


# ## 5. Save Validation Results

# In[ ]:


# Save validation summary
validation_summary = {
    'validation_date': datetime.now().strftime('%Y-%m-%d'),
    'training_period': '2019-2023',
    'holdout_period': '2024',
    'training_samples': len(train_data),
    'holdout_samples': len(holdout_2024),
    'c_index_2024': c_index_2024,
    'training_injury_rate': train_data['event'].mean(),
    'holdout_injury_rate': holdout_2024['event'].mean(),
    'projection_samples_2025': len(projections_2025)
}

import json
with open('../data/processed/validation_results.json', 'w') as f:
    json.dump(validation_summary, f, indent=2)

print("✅ Validation complete!")
print(f"External validation C-index: {c_index_2024:.3f}")
print(f"2025 projections ready for {len(projections_2025)} pitchers")

