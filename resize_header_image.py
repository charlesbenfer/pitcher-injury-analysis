"""
Resize the header image to 75% of its current size
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create header image at 75% size (was 16x8, now 12x6)
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# 1. Survival curve
ax = axes[0, 0]
times = np.linspace(0, 180, 100)
low_risk = np.exp(-times/200)
high_risk = np.exp(-times/80)
ax.plot(times, low_risk, 'g-', linewidth=2.5, label='Low Risk')
ax.plot(times, high_risk, 'r-', linewidth=2.5, label='High Risk')
ax.fill_between(times, low_risk, alpha=0.3, color='green')
ax.fill_between(times, high_risk, alpha=0.3, color='red')
ax.set_xlabel('Days', fontsize=10)
ax.set_ylabel('Injury-Free Probability', fontsize=10)
ax.set_title('Survival Curves by Risk Category', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Feature importance
ax = axes[0, 1]
features = ['Age', 'Games\nPlayed', 'ERA', 'Innings\nPitched', 'Veteran\nStatus', 'WAR', 'Workload']
importance = [0.22, 0.19, 0.15, 0.14, 0.12, 0.10, 0.08]
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(features)))
bars = ax.barh(features, importance, color=colors, edgecolor='black')
ax.set_xlabel('Feature Importance', fontsize=10)
ax.set_title('Risk Factor Importance', fontsize=12, fontweight='bold')
ax.set_xlim(0, 0.3)
ax.tick_params(axis='y', labelsize=9)

# 3. Bayesian uncertainty
ax = axes[0, 2]
x = np.linspace(-3, 3, 100)
prior = np.exp(-x**2/8) / np.sqrt(8*np.pi)
posterior = np.exp(-(x-0.5)**2/2) / np.sqrt(2*np.pi)
ax.plot(x, prior, 'b--', linewidth=2, label='Prior Belief', alpha=0.7)
ax.plot(x, posterior, 'r-', linewidth=2.5, label='Posterior (After Data)')
ax.fill_between(x, posterior, alpha=0.3, color='red')
ax.set_xlabel('Effect Size', fontsize=10)
ax.set_ylabel('Probability Density', fontsize=10)
ax.set_title('Bayesian Learning Process', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Risk matrix heatmap
ax = axes[1, 0]
risk_matrix = np.array([
    [0.25, 0.30, 0.35, 0.42],
    [0.28, 0.34, 0.40, 0.48],
    [0.32, 0.38, 0.45, 0.55],
    [0.38, 0.45, 0.52, 0.65]
])
im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.2, vmax=0.7)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=9)
ax.set_yticklabels(['Young', 'Prime', 'Veteran', 'Aging'], fontsize=9)
ax.set_xlabel('Workload Quartile', fontsize=10)
ax.set_ylabel('Age Category', fontsize=10)
ax.set_title('Injury Risk by Age & Workload', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(4):
    for j in range(4):
        text = ax.text(j, i, f'{risk_matrix[i, j]:.0%}',
                      ha="center", va="center", 
                      color="white" if risk_matrix[i, j] > 0.45 else "black",
                      fontsize=9, fontweight='bold')

# 5. Model performance metrics
ax = axes[1, 1]
metrics = ['C-Index', 'Brier\nScore', 'Calibration', 'Coverage']
values = [0.607, 0.771, 0.92, 0.742]  # Normalized for display
colors = ['#2E8B57' if v >= 0.6 else '#FF8C00' for v in values]
bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Performance Score', fontsize=10)
ax.set_title('Model Validation Metrics', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', labelsize=9)

# Add value labels
for bar, val, metric in zip(bars, values, ['0.607', '0.229', 'Good', '74.2%']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           metric, ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. 2025 projections preview
ax = axes[1, 2]
teams = ['Yankees', 'Dodgers', 'Padres', 'Astros', 'Braves']
high_risk_counts = [5, 4, 6, 3, 4]
colors = ['#003087', '#005A9C', '#2F241D', '#002D62', '#CE1141']
bars = ax.bar(teams, high_risk_counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('High-Risk Pitchers', fontsize=10)
ax.set_title('2025 High-Risk Pitchers by Team', fontsize=12, fontweight='bold')
ax.set_ylim(0, 8)
ax.tick_params(axis='x', labelsize=9, rotation=45)

# Add count labels
for bar, count in zip(bars, high_risk_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
           str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add main title and subtitle (smaller fonts)
fig.suptitle('Bayesian Survival Analysis for MLB Pitcher Injury Prediction', 
             fontsize=16, fontweight='bold', y=1.02)
fig.text(0.5, 0.98, 'Predicting When (Not Just If) Pitchers Will Get Injured', 
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.savefig('/home/charlesbenfer/charlesbenfer.github.io/assets/img/pitcher_injury_header.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Header image resized to 75% (12x6 inches) and saved to assets/img/pitcher_injury_header.png")