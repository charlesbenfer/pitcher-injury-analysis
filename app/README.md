# ⚾ MLB Pitcher Injury Risk Dashboard

## 🚀 Quick Start

### Option 1: Launch from project root
```bash
python launch_dashboard.py
```

### Option 2: Direct streamlit command
```bash
cd app/
streamlit run pitcher_risk_app.py
```

## 📱 Dashboard Features

### 🔍 Individual Risk Tab
- **Select any pitcher** from dropdown or use random button
- **Real-time risk assessment** with color-coded risk levels
- **Risk timeline** showing 30-day, 60-day, 4-month, and season injury probabilities
- **Clinical recommendations** based on risk level
- **Key statistics** and performance metrics
- **Actual outcomes** for validation

### 📊 Team Overview Tab
- **Risk distribution** pie chart for entire roster
- **High-risk pitchers** table with sortable columns
- **Age vs Risk** interactive scatter plot
- **Team statistics** by risk category

### 📈 Risk Analysis Tab
- **Model performance** metrics and accuracy
- **Risk gradient validation** showing injury rates by category
- **Interactive visualizations** of model effectiveness

### 📋 Reports Tab
- **Generate comprehensive reports** for medical staff
- **Download reports** as Markdown files
- **Export data** as CSV for spreadsheet analysis

## 🎯 Risk Categories

- **🟢 Low Risk**: 21.7% injury rate - Standard monitoring
- **🟡 Moderate Risk**: 30.0% injury rate - Enhanced monitoring  
- **🟠 High Risk**: 47.1% injury rate - Weekly assessments
- **🔴 Very High Risk**: 58.5% injury rate - Daily monitoring

## 📊 Model Details

- **Algorithm**: Bayesian Weibull Accelerated Failure Time (AFT) model
- **Performance**: C-index = 0.607 (good discrimination)
- **Training Data**: 1,284 pitcher-seasons with 523 injury events
- **Features**: Age, workload, performance metrics from previous season
- **Validation**: Realistic risk stratification with monotonic injury rates

## 🏥 Clinical Applications

- **Pre-season planning**: Identify at-risk pitchers before season starts
- **In-season monitoring**: Real-time risk assessment during season
- **Medical decision support**: Evidence-based recommendations for medical staff
- **Roster management**: Inform trading, assignment, and workload decisions
- **Injury prevention**: Targeted interventions based on risk scores

## 🔧 Technical Requirements

- Python 3.8+
- Streamlit 1.28+
- Plotly 5.15+
- Pandas 2.0+
- NumPy 1.24+

## 📁 Data Requirements

The dashboard expects the dataset at:
```
data/processed/survival_dataset_lagged_enhanced.csv
```

## 🚨 Usage Notes

- **Select season** from sidebar to filter data
- **Adjust minimum games** threshold to focus on active pitchers
- **Risk scores update** automatically when filters change
- **Download reports** for offline analysis and medical team distribution
- **Color coding** throughout dashboard for quick risk identification

## 🎯 Future Enhancements

- Real-time data integration
- Biomechanical metrics incorporation  
- Multi-team comparison views
- Mobile-responsive design
- API endpoints for team system integration

---

**Developed for MLB teams to prevent pitcher injuries through data-driven risk assessment**