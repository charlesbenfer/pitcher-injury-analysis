#!/bin/bash

# Data cleanup script for pitcher injury analysis
# Keeps only essential files

echo "Creating backup of current data directory..."
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)/

echo "Files to be deleted:"
echo "==================="

# List files that will be deleted (excluding essential ones)
find data/processed/ -name "*.csv" -not -name "survival_dataset_lagged.csv" -not -name "enhanced_survival_dataset_*.csv"

echo ""
echo "Files to keep:"
echo "=============="
echo "data/processed/survival_dataset_lagged.csv (original clean dataset)"
echo "../data/raw/pitcher_injuries_2018_2024.csv (new comprehensive dataset)"
echo "../data/processed/enhanced_survival_dataset_20250902.csv (integrated dataset)"

echo ""
read -p "Do you want to delete the unnecessary files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting unnecessary data files..."
    
    # Delete old processed files (keeping essential ones)
    find data/processed/ -name "*.csv" -not -name "survival_dataset_lagged.csv" -not -name "enhanced_survival_dataset_*.csv" -delete
    
    # Clean up any old raw files in current directory
    rm -f data/raw/*.csv 2>/dev/null
    
    echo "Cleanup completed!"
    echo "Remaining data files:"
    find data/ -name "*.csv" -type f
else
    echo "Cleanup cancelled."
fi