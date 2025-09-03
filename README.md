# MLB Pitcher Injury Prediction with Bayesian Survival Analysis

This repository contains the code and analysis for predicting MLB pitcher injuries using Bayesian survival analysis methods.

## Blog Post

For a detailed walkthrough of the methodology and findings, see the full blog post:
[Predicting Pitcher Injuries with Bayesian Survival Analysis](https://charlesbenfer.github.io/2025/09/02/bayesian-pitcher-injury.html)

## Project Overview

- **Objective**: Predict when (not just if) MLB pitchers will get injured
- **Method**: Bayesian Weibull Accelerated Failure Time (AFT) models
- **Performance**: C-index of 0.607 (0.592 on 2024 holdout data)
- **Key Finding**: Simple linear models outperformed complex non-linear approaches

## Repository Structure

```
pitcher_injury_analysis/
├── notebooks/
│   └── bayesian_survival_analysis.ipynb  # Main analysis notebook
├── app/
│   └── pitcher_risk_app.py               # Streamlit dashboard
├── scripts/
│   ├── validate_with_holdout_2024.py     # Model validation
│   └── evaluate_elbow_model.py           # Tommy John specific model
├── data/
│   ├── raw/                              # Original MLB data
│   └── processed/                        # Processed datasets
└── requirements.txt                       # Python dependencies
```

## Key Features

- Bayesian survival analysis using PyMC
- Production-ready risk scoring system
- Interactive Streamlit dashboard for risk assessment
- 2025 season projections for all active pitchers
- Team-level risk analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/pitcher-injury-analysis.git
cd pitcher-injury-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

### Main Bayesian Analysis
```bash
jupyter notebook notebooks/bayesian_survival_analysis.ipynb
```

### Interactive Dashboard
```bash
streamlit run app/pitcher_risk_app.py
```

### Model Validation
```bash
python scripts/validate_with_holdout_2024.py
```

## Model Performance

- **Training (2019-2023)**: C-index 0.607
- **Validation (2024)**: C-index 0.592
- **Risk Stratification**:
  - Low Risk: 28.8% injury rate
  - Moderate Risk: 34.6% injury rate
  - High Risk: 39.7% injury rate
  - Very High Risk: 49.1% injury rate

## Technologies Used

- **PyMC**: Bayesian modeling framework
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Model evaluation metrics
- **Matplotlib/Seaborn**: Visualizations

## Future Improvements

The model would benefit significantly from biomechanical data:
- Release point consistency
- Velocity trends
- Spin rate changes
- Arm slot variations

These features are likely more predictive than performance statistics alone.

## Citation

If you use this code or methodology, please cite:
```
Benfer, C. (2025). Predicting Pitcher Injuries with Bayesian Survival Analysis.
https://charlesbenfer.github.io/2025/09/02/bayesian-pitcher-injury.html
```

## License

MIT License - See LICENSE file for details

## Contact

Charles Benfer - [LinkedIn](https://www.linkedin.com/in/charles-benfer-b6508a161/)

Project Link: [https://github.com/charlesbenfer/pitcher-injury-analysis](https://github.com/charlesbenfer/pitcher-injury-analysis)
