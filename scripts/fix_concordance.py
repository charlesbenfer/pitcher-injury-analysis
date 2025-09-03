"""
Fix to apply to the concordance calculation in the Bayesian survival notebook
"""

# CURRENT CODE (wrong for AFT models):
"""
# Higher risk should have shorter survival time
if (risk_scores[i] > risk_scores[j] and times[i] < times[j]) or \
   (risk_scores[i] < risk_scores[j] and times[i] > times[j]):
    concordant += 1
"""

# FIXED CODE (correct for AFT models):
"""
# In AFT models: Higher linear predictor = LONGER survival = LOWER risk
# So we need to flip the relationship
if (risk_scores[i] > risk_scores[j] and times[i] > times[j]) or \
   (risk_scores[i] < risk_scores[j] and times[i] < times[j]):
    concordant += 1
"""

print("Manual fix needed in bayesian_survival_analysis.ipynb:")
print("In the compute_concordance_index function, change:")
print("OLD:")
print("    if (risk_scores[i] > risk_scores[j] and times[i] < times[j]) or \\")
print("       (risk_scores[i] < risk_scores[j] and times[i] > times[j]):")
print("")
print("NEW:")  
print("    if (risk_scores[i] > risk_scores[j] and times[i] > times[j]) or \\")
print("       (risk_scores[i] < risk_scores[j] and times[i] < times[j]):")
print("")
print("This flips the logic because AFT models predict survival TIME, not hazard.")