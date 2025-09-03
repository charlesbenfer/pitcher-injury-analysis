"""
Fix for the spline key error in the non-linear age notebook
"""

# The issue is that best_approach = 'spline_5knots' 
# but spline_features uses keys like 'cubic_5knots'

# Here's the mapping fix:
spline_key_mapping = {
    'spline_4knots': 'cubic_4knots',
    'spline_5knots': 'cubic_5knots',
    'spline_6knots': 'cubic_6knots'
}

print("🔧 SPLINE KEY MAPPING FIX")
print("=" * 30)
print("Replace this line in the notebook:")
print("final_age_features = spline_features[best_approach]")
print()
print("With this:")
print("if 'spline' in best_approach:")
print("    spline_key = best_approach.replace('spline_', 'cubic_')")
print("    final_age_features = spline_features[spline_key]")
print()
print("This maps:")
for old_key, new_key in spline_key_mapping.items():
    print(f"  {old_key} → {new_key}")