"""
Interactive Web Dashboard for MLB Pitcher Injury Risk Assessment
Real dashboard where you can select pitchers and get their risk information
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="MLB Pitcher Injury Risk Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PitcherRiskScorer:
    """Production-ready pitcher injury risk scoring system"""
    
    def __init__(self):
        # Calibrated risk quartiles and feature weights
        self.risk_quartiles = [-5.154, -5.008, -4.931]
        self.beta_0 = 5.0
        self.alpha = 2.0
        
        self.feature_weights = {
            'age_prev': -0.01,
            'g_prev': 0.0075,
            'veteran_prev': 0.004,
            'era_prev': -0.004,
            'ip_prev': 0.0004,
            'war_prev': 0.002,
            'high_workload_prev': -0.002
        }
    
    def calculate_risk_score(self, pitcher_stats):
        """Calculate comprehensive risk assessment for a pitcher"""
        # Calculate linear predictor
        linear_pred = self.beta_0
        for feature, weight in self.feature_weights.items():
            if feature in pitcher_stats:
                linear_pred += weight * pitcher_stats[feature]
        
        risk_score = -linear_pred
        
        # Risk categorization
        if risk_score <= self.risk_quartiles[0]:
            risk_category = "Low"
            color = "🟢"
            alert_level = 0
            color_hex = "#2E8B57"
        elif risk_score <= self.risk_quartiles[1]:
            risk_category = "Moderate"
            color = "🟡"
            alert_level = 1
            color_hex = "#FFD700"
        elif risk_score <= self.risk_quartiles[2]:
            risk_category = "High"
            color = "🟠"
            alert_level = 2
            color_hex = "#FF8C00"
        else:
            risk_category = "Very High"
            color = "🔴"
            alert_level = 3
            color_hex = "#DC143C"
        
        # Survival probabilities
        scale = np.exp(linear_pred)
        survival_30 = np.exp(-(30/scale)**self.alpha)
        survival_60 = np.exp(-(60/scale)**self.alpha)
        survival_120 = np.exp(-(120/scale)**self.alpha)
        survival_180 = np.exp(-(180/scale)**self.alpha)
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'color_code': color,
            'color_hex': color_hex,
            'alert_level': alert_level,
            'injury_risk_30': (1 - survival_30) * 100,
            'injury_risk_60': (1 - survival_60) * 100,
            'injury_risk_120': (1 - survival_120) * 100,
            'injury_risk_season': (1 - survival_180) * 100
        }
    
    def get_recommendations(self, risk_category):
        """Get clinical recommendations"""
        recommendations = {
            "Low": [
                "✅ Continue standard training schedule",
                "📅 Annual physical assessment",
                "📊 Monitor workload during high-stress periods"
            ],
            "Moderate": [
                "🔍 Enhanced monitoring protocols",
                "⚖️ Consider workload management",
                "🏥 Quarterly biomechanical assessments",
                "💪 Focus on injury prevention exercises"
            ],
            "High": [
                "🚨 MANDATORY biomechanical evaluation",
                "📉 Significant workload restrictions",
                "👨‍⚕️ Weekly medical staff monitoring",
                "🔧 Consider mechanical adjustments",
                "🛡️ Enhanced recovery protocols"
            ],
            "Very High": [
                "🆘 IMMEDIATE comprehensive medical evaluation",
                "⛔ Major workload reduction required",
                "📊 Daily monitoring protocols",
                "🩻 Mandatory imaging assessment",
                "👨‍⚕️ Sports medicine specialist consultation"
            ]
        }
        return recommendations.get(risk_category, [])

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare pitcher data"""
    try:
        df = pd.read_csv('data/processed/survival_dataset_lagged_enhanced.csv')
        
        # Available features
        feature_cols = [
            'age_prev', 'w_prev', 'l_prev', 'era_prev', 'g_prev', 'gs_prev', 'ip_prev',
            'h_prev', 'r_prev', 'er_prev', 'hr_prev', 'bb_prev', 'so_prev', 'whip_prev',
            'k_per_9_prev', 'bb_per_9_prev', 'hr_per_9_prev', 'fip_prev', 'war_prev',
            'high_workload_prev', 'veteran_prev', 'high_era_prev'
        ]
        
        available_features = [f for f in feature_cols if f in df.columns]
        
        return df, available_features
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure the data file is in the correct location.")
        return None, None

def main():
    """Main dashboard application"""
    
    # Header
    st.title("⚾ MLB Pitcher Injury Risk Dashboard")
    st.markdown("**Real-time injury risk assessment based on Bayesian survival analysis (C-index: 0.607)**")
    
    # Load data
    df, available_features = load_data()
    if df is None:
        st.stop()
    
    # Initialize risk scorer
    risk_scorer = PitcherRiskScorer()
    
    # Add team information (since it's not in the original data, we'll assign teams)
    @st.cache_data
    @st.cache_data
    def load_team_mapping():
        """Load real MLB team mapping from current roster data"""
        try:
            # Try to use the current accurate mapping first
            team_mapping = pd.read_csv('data/processed/player_team_mapping_current.csv')
            # Handle 'Multiple Teams' players as 'Traded'
            team_mapping['team_name'] = team_mapping['team_name'].replace('Multiple Teams', 'Traded')
            return dict(zip(team_mapping['player_name'], team_mapping['team_name']))
        except FileNotFoundError:
            # Fallback to old mapping if current doesn't exist
            try:
                team_mapping = pd.read_csv('data/processed/player_team_mapping.csv')
                return dict(zip(team_mapping['player_name'], team_mapping['team_name']))
            except FileNotFoundError:
                st.error("Team mapping file not found. Please run fetch_current_teams.py first.")
                return {}
    
    def add_team_data(data):
        """Add MLB team assignments to pitcher data using real team mapping"""
        team_mapping = load_team_mapping()
        
        if not team_mapping:
            st.warning("No team mapping available. Team functionality disabled.")
            return data
        
        data_copy = data.copy()
        # Map players to their real teams
        data_copy['team'] = data_copy['player_name'].map(team_mapping)
        
        # Handle unmapped players - mark as 'Traded'
        unmapped_count = data_copy['team'].isna().sum()
        if unmapped_count > 0:
            st.sidebar.info(f"Note: {unmapped_count} players marked as 'Traded' (mid-season transactions)")
            # Mark unmapped players as traded
            data_copy['team'] = data_copy['team'].fillna('Unknown Team')
        
        return data_copy
    
    df = add_team_data(df)
    
    # Sidebar filters
    st.sidebar.header("🎯 Dashboard Controls")
    
    # Season filter
    seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.sidebar.selectbox("📅 Select Season", seasons, index=0)
    
    # Filter data by season
    season_data = df[df['season'] == selected_season].copy()
    
    # Team filter - include 'Traded' for players without team assignments
    teams_with_data = sorted([team for team in season_data['team'].unique() if pd.notna(team) and team != 'Unknown Team'])
    
    # Check if there are traded players (Unknown Team)
    traded_count = len(season_data[season_data['team'] == 'Unknown Team'])
    team_options = ["All Teams"] + teams_with_data
    if traded_count > 0:
        team_options.append("Traded")
    
    selected_team = st.sidebar.selectbox("⚾ Select Team", team_options, index=0)
    
    # Filter by team
    if selected_team == "All Teams":
        team_data = season_data.copy()
        st.sidebar.markdown(f"**Viewing: All MLB Teams**")
        st.sidebar.markdown(f"**Teams with data: {len(teams_with_data)}**")
        if traded_count > 0:
            st.sidebar.markdown(f"**Traded players: {traded_count}**")
    elif selected_team == "Traded":
        team_data = season_data[season_data['team'] == 'Unknown Team'].copy()
        st.sidebar.markdown(f"**Viewing: Traded Players**")
        st.sidebar.markdown(f"**Players: {len(team_data)}**")
        st.sidebar.info("These players likely changed teams mid-season")
    else:
        team_data = season_data[season_data['team'] == selected_team].copy()
        st.sidebar.markdown(f"**Viewing: {selected_team}**")
        if len(team_data) == 0:
            st.sidebar.warning(f"No data found for {selected_team}")
    
    # Minimum games filter
    min_games = st.sidebar.slider("🎮 Minimum Games (Previous Season)", 0, 50, 10)
    filtered_data = team_data[team_data['g_prev'] >= min_games].copy()
    
    st.sidebar.markdown(f"**Active Pitchers: {len(filtered_data)}**")
    if selected_team != "All Teams":
        st.sidebar.markdown(f"**Team Pitchers: {len(team_data)}**")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Individual Risk", "📊 Team Overview", "📈 Risk Analysis", "📋 Reports", "🚀 2025 Projections"])
    
    with tab1:
        st.header("🔍 Individual Pitcher Risk Assessment")
        
        # Pitcher selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pitcher_names = sorted(filtered_data['player_name'].unique())
            selected_pitcher = st.selectbox("Select a Pitcher:", pitcher_names)
        
        with col2:
            if st.button("🎲 Random Pitcher", type="secondary"):
                selected_pitcher = np.random.choice(pitcher_names)
                st.rerun()
        
        if selected_pitcher:
            # Get pitcher data
            pitcher_data = filtered_data[filtered_data['player_name'] == selected_pitcher].iloc[0]
            pitcher_stats = pitcher_data[available_features].to_dict()
            
            # Calculate risk
            risk_result = risk_scorer.calculate_risk_score(pitcher_stats)
            recommendations = risk_scorer.get_recommendations(risk_result['risk_category'])
            
            # Display risk assessment
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"### {selected_pitcher} ({selected_season})")
                
                # Risk level with color
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {risk_result['color_hex']}20; border: 2px solid {risk_result['color_hex']}'>
                    <h1 style='color: {risk_result['color_hex']}; margin: 0;'>{risk_result['color_code']} {risk_result['risk_category'].upper()} RISK</h1>
                    <p style='font-size: 18px; margin: 5px 0;'>Alert Level: {risk_result['alert_level']}</p>
                    <p style='font-size: 16px; margin: 0;'>Risk Score: {risk_result['risk_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk timeline
            st.subheader("⏰ Injury Risk Timeline")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("30 Days", f"{risk_result['injury_risk_30']:.1f}%")
            with col2:
                st.metric("60 Days", f"{risk_result['injury_risk_60']:.1f}%")
            with col3:
                st.metric("4 Months", f"{risk_result['injury_risk_120']:.1f}%")
            with col4:
                st.metric("Full Season", f"{risk_result['injury_risk_season']:.1f}%")
            
            # Key statistics
            st.subheader("📈 Key Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Performance Metrics:**")
                st.write(f"• Age: {pitcher_data['age_prev']:.0f}")
                st.write(f"• Games: {pitcher_data['g_prev']:.0f}")
                st.write(f"• ERA: {pitcher_data['era_prev']:.2f}")
                st.write(f"• Innings: {pitcher_data['ip_prev']:.1f}")
                st.write(f"• WAR: {pitcher_data.get('war_prev', 'N/A')}")
            
            with col2:
                st.write("**Risk Factors:**")
                st.write(f"• Veteran Status: {'Yes' if pitcher_data.get('veteran_prev', 0) else 'No'}")
                st.write(f"• High Workload: {'Yes' if pitcher_data.get('high_workload_prev', 0) else 'No'}")
                st.write(f"• High ERA: {'Yes' if pitcher_data.get('high_era_prev', 0) else 'No'}")
            
            # Recommendations
            st.subheader("📋 Clinical Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
            
            # Actual outcome (if available)
            if 'event' in pitcher_data:
                actual_outcome = "Injury" if pitcher_data['event'] == 1 else "No Injury"
                actual_time = pitcher_data['time_to_event']
                
                st.subheader("✅ Actual Outcome")
                outcome_color = "red" if pitcher_data['event'] == 1 else "green"
                st.markdown(f"**<span style='color: {outcome_color}'>{actual_outcome}</span>** (at {actual_time} days)", unsafe_allow_html=True)
    
    with tab2:
        if selected_team == "All Teams":
            st.header("📊 League Risk Overview")
        elif selected_team == "Traded":
            st.header("🔄 Traded Players - Risk Overview")
        else:
            st.header(f"📊 {selected_team} - Risk Overview")
        
        # Calculate risk for all pitchers
        @st.cache_data
        def calculate_team_risks(data):
            risks = []
            for _, pitcher in data.iterrows():
                pitcher_stats = pitcher[available_features].to_dict()
                risk_result = risk_scorer.calculate_risk_score(pitcher_stats)
                
                risks.append({
                    'player_name': pitcher['player_name'],
                    'risk_category': risk_result['risk_category'],
                    'risk_score': risk_result['risk_score'],
                    'alert_level': risk_result['alert_level'],
                    'injury_risk_season': risk_result['injury_risk_season'],
                    'age': pitcher['age_prev'],
                    'era': pitcher['era_prev'],
                    'games': pitcher['g_prev'],
                    'color_hex': risk_result['color_hex'],
                    'actual_injury': pitcher.get('event', 0)
                })
            
            return pd.DataFrame(risks).sort_values('risk_score', ascending=False)
        
        team_risks = calculate_team_risks(filtered_data)
        
        # Team-specific header info
        if selected_team == "Traded":
            st.info(f"🔄 **Traded Players Analysis** - {len(filtered_data)} pitchers who changed teams mid-season (min {min_games} games)")
        elif selected_team != "All Teams":
            st.info(f"👀 **{selected_team} Roster Analysis** - {len(filtered_data)} active pitchers with ≥{min_games} games")
            
            # Add team vs league comparison (only if we have sufficient data)
            try:
                league_data = season_data[season_data['g_prev'] >= min_games]
                # Only include teams with actual data for league comparison
                league_data_with_teams = league_data[league_data['team'] != 'Unknown Team']
                
                if len(league_data_with_teams) > 0:
                    league_risks = calculate_team_risks(league_data_with_teams)
                    
                    team_high_risk_pct = (team_risks['alert_level'] >= 2).mean() * 100
                    league_high_risk_pct = (league_risks['alert_level'] >= 2).mean() * 100
                    
                    team_avg_risk = team_risks['injury_risk_season'].mean()
                    league_avg_risk = league_risks['injury_risk_season'].mean()
                else:
                    # Fallback if no league data available
                    league_risks = team_risks
                    team_high_risk_pct = (team_risks['alert_level'] >= 2).mean() * 100
                    league_high_risk_pct = team_high_risk_pct
                    team_avg_risk = team_risks['injury_risk_season'].mean()
                    league_avg_risk = team_avg_risk
            except Exception as e:
                st.sidebar.error(f"Error calculating league comparison: {str(e)[:50]}...")
                league_risks = team_risks
                team_high_risk_pct = (team_risks['alert_level'] >= 2).mean() * 100
                league_high_risk_pct = team_high_risk_pct
                team_avg_risk = team_risks['injury_risk_season'].mean()
                league_avg_risk = team_avg_risk
            
            col1, col2, col3 = st.columns(3)
            with col1:
                delta_high_risk = team_high_risk_pct - league_high_risk_pct
                st.metric(
                    "High Risk Pitchers", 
                    f"{team_high_risk_pct:.1f}%",
                    delta=f"{delta_high_risk:+.1f}% vs League"
                )
            with col2:
                delta_avg_risk = team_avg_risk - league_avg_risk
                st.metric(
                    "Avg Season Risk", 
                    f"{team_avg_risk:.1f}%",
                    delta=f"{delta_avg_risk:+.1f}% vs League"
                )
            with col3:
                try:
                    team_rankings = league_risks.groupby(league_data['team'])['injury_risk_season'].mean().sort_values(ascending=False)
                    if selected_team in team_rankings.index:
                        team_rank = team_rankings.index.tolist().index(selected_team) + 1
                        total_teams = len(team_rankings)
                        st.metric(
                            "League Rank", 
                            f"#{team_rank} of {total_teams}",
                            delta=f"{'Higher' if team_rank <= total_teams//2 else 'Lower'} Risk"
                        )
                    else:
                        st.metric(
                            "League Rank", 
                            "Not Available",
                            delta="Insufficient data"
                        )
                except Exception as e:
                    st.metric(
                        "League Rank", 
                        "Error",
                        delta="Data issue"
                    )
        else:
            traded_in_data = len(filtered_data[filtered_data['team'] == 'Unknown Team'])
            teams_in_data = len([team for team in filtered_data['team'].unique() if team != 'Unknown Team'])
            st.info(f"🌎 **League-wide Analysis** - {len(filtered_data)} active pitchers ({teams_in_data} teams + {traded_in_data} traded players)")
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_counts = team_risks['risk_category'].value_counts()
            
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={
                    'Low': '#2E8B57',
                    'Moderate': '#FFD700', 
                    'High': '#FF8C00',
                    'Very High': '#DC143C'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Risk Statistics")
            for category in ['Low', 'Moderate', 'High', 'Very High']:
                count = len(team_risks[team_risks['risk_category'] == category])
                pct = count / len(team_risks) * 100
                color_map = {'Low': '🟢', 'Moderate': '🟡', 'High': '🟠', 'Very High': '🔴'}
                st.write(f"{color_map[category]} **{category}**: {count} pitchers ({pct:.1f}%)")
        
        # High-risk pitchers table
        st.subheader("🚨 High-Risk Pitchers")
        high_risk = team_risks[team_risks['alert_level'] >= 2]
        
        if len(high_risk) > 0:
            # Create display dataframe
            display_df = high_risk[['player_name', 'risk_category', 'injury_risk_season', 'age', 'era', 'games']].copy()
            display_df['injury_risk_season'] = display_df['injury_risk_season'].round(1)
            display_df.columns = ['Pitcher', 'Risk Level', 'Season Risk (%)', 'Age', 'ERA', 'Games']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No high-risk pitchers identified in current filter.")
        
        # Team roster table (only for specific team selection)
        if selected_team != "All Teams":
            st.subheader(f"📋 {selected_team} - Complete Roster")
            
            # Create comprehensive roster display
            roster_df = team_risks[['player_name', 'risk_category', 'injury_risk_season', 'age', 'era', 'games', 'alert_level']].copy()
            roster_df['injury_risk_season'] = roster_df['injury_risk_season'].round(1)
            roster_df = roster_df.sort_values(['alert_level', 'injury_risk_season'], ascending=[False, False])
            
            # Add color coding
            def color_risk_level(val):
                colors = {'Low': '#2E8B57', 'Moderate': '#FFD700', 'High': '#FF8C00', 'Very High': '#DC143C'}
                return f'background-color: {colors.get(val, "#FFFFFF")}20'
            
            roster_display = roster_df[['player_name', 'risk_category', 'injury_risk_season', 'age', 'era', 'games']].copy()
            roster_display.columns = ['Pitcher', 'Risk Level', 'Season Risk (%)', 'Age', 'ERA', 'Games']
            
            st.dataframe(
                roster_display.style.applymap(color_risk_level, subset=['Risk Level']),
                use_container_width=True,
                height=400
            )
            
            # Download team roster CSV
            csv_data = roster_display.to_csv(index=False)
            st.download_button(
                label=f"💾 Download {selected_team} Roster CSV",
                data=csv_data,
                file_name=f"{selected_team.replace(' ', '_')}_pitcher_risks_{selected_season}.csv",
                mime="text/csv"
            )
        
        # Age vs Risk scatter plot with team context
        st.subheader("📈 Age vs Risk Analysis")
        
        if selected_team != "All Teams":
            # Show team vs league comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{selected_team} Pitchers**")
                fig_scatter_team = px.scatter(
                    team_risks,
                    x='age',
                    y='injury_risk_season',
                    color='risk_category',
                    size='games',
                    hover_data=['player_name', 'era'],
                    color_discrete_map={
                        'Low': '#2E8B57',
                        'Moderate': '#FFD700',
                        'High': '#FF8C00', 
                        'Very High': '#DC143C'
                    },
                    title=f"{selected_team} - Age vs Risk"
                )
                fig_scatter_team.update_layout(height=400)
                st.plotly_chart(fig_scatter_team, use_container_width=True)
            
            with col2:
                st.write("**League Average**")
                league_data = season_data[season_data['g_prev'] >= min_games]
                league_risks = calculate_team_risks(league_data)
                
                fig_scatter_league = px.scatter(
                    league_risks,
                    x='age',
                    y='injury_risk_season',
                    color='risk_category',
                    size='games',
                    opacity=0.6,
                    hover_data=['player_name'],
                    color_discrete_map={
                        'Low': '#2E8B57',
                        'Moderate': '#FFD700',
                        'High': '#FF8C00', 
                        'Very High': '#DC143C'
                    },
                    title="League - Age vs Risk"
                )
                fig_scatter_league.update_layout(height=400)
                st.plotly_chart(fig_scatter_league, use_container_width=True)
        else:
            fig_scatter = px.scatter(
                team_risks,
                x='age',
                y='injury_risk_season',
                color='risk_category',
                size='games',
                hover_data=['player_name', 'era'],
                color_discrete_map={
                    'Low': '#2E8B57',
                    'Moderate': '#FFD700',
                    'High': '#FF8C00', 
                    'Very High': '#DC143C'
                },
                title="Age vs Season Injury Risk"
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.header("📈 Risk Analysis")
        
        if team_risks is not None:
            # Model performance
            st.subheader("🎯 Model Performance")
            
            injured = team_risks[team_risks['actual_injury'] == 1]
            high_risk_injured = injured[injured['alert_level'] >= 2]
            
            if len(injured) > 0:
                accuracy = len(high_risk_injured) / len(injured) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Injuries", len(injured))
                with col2:
                    st.metric("High-Risk Injuries Identified", f"{len(high_risk_injured)}/{len(injured)}")
                with col3:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
            
            # Risk gradient validation
            st.subheader("📊 Risk Gradient Validation")
            
            gradient_data = []
            for category in ['Low', 'Moderate', 'High', 'Very High']:
                cat_data = team_risks[team_risks['risk_category'] == category]
                if len(cat_data) > 0:
                    injury_rate = cat_data['actual_injury'].mean() * 100
                    gradient_data.append({
                        'Risk Category': category,
                        'Injury Rate (%)': injury_rate,
                        'Count': len(cat_data)
                    })
            
            gradient_df = pd.DataFrame(gradient_data)
            
            fig_bar = px.bar(
                gradient_df,
                x='Risk Category',
                y='Injury Rate (%)',
                color='Risk Category',
                color_discrete_map={
                    'Low': '#2E8B57',
                    'Moderate': '#FFD700',
                    'High': '#FF8C00',
                    'Very High': '#DC143C'
                },
                title="Actual Injury Rate by Risk Category"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab4:
        st.header("📋 Risk Reports")
        
        # Generate downloadable report
        if st.button("📄 Generate Team Risk Report"):
            
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            report_content = f"""
# MLB PITCHER INJURY RISK ASSESSMENT REPORT
Generated: {report_date}
Season: {selected_season}
Model: Bayesian Survival Analysis (C-index: 0.607)

## EXECUTIVE SUMMARY
- Total Pitchers Analyzed: {len(team_risks)}
- High-Risk Pitchers (Alert Level 2+): {len(team_risks[team_risks['alert_level'] >= 2])}
- Average Season Injury Risk: {team_risks['injury_risk_season'].mean():.1f}%

## RISK DISTRIBUTION
"""
            
            risk_counts = team_risks['risk_category'].value_counts()
            for category in ['Very High', 'High', 'Moderate', 'Low']:
                count = risk_counts.get(category, 0)
                pct = count / len(team_risks) * 100
                report_content += f"- {category}: {count} pitchers ({pct:.1f}%)\n"
            
            report_content += "\n## HIGH-RISK PITCHERS REQUIRING ATTENTION\n"
            
            high_risk = team_risks[team_risks['alert_level'] >= 2].head(10)
            for _, pitcher in high_risk.iterrows():
                report_content += f"\n### {pitcher['player_name']} - {pitcher['risk_category']} Risk\n"
                report_content += f"- Season Injury Risk: {pitcher['injury_risk_season']:.1f}%\n"
                report_content += f"- Age: {pitcher['age']:.0f}, ERA: {pitcher['era']:.2f}\n"
            
            st.text_area("Generated Report:", report_content, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"pitcher_risk_report_{selected_season}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        # CSV download
        if st.button("📊 Export Data as CSV"):
            csv_data = team_risks.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"pitcher_risks_{selected_season}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab5:
        st.header("🚀 2025 Season Projections")
        
        # Load 2025 projections
        @st.cache_data
        def load_2025_projections():
            try:
                projections = pd.read_csv('data/processed/pitcher_projections_2025.csv')
                # Add team data if available
                team_mapping = load_team_mapping()
                if team_mapping:
                    projections['team'] = projections['player_name'].map(team_mapping)
                    projections = projections.dropna(subset=['team'])
                return projections
            except FileNotFoundError:
                st.error("🚨 2025 projections not found. Please run validation analysis first.")
                return pd.DataFrame()
        
        projections_2025 = load_2025_projections()
        
        if len(projections_2025) > 0:
            st.info(f"🔮 **Forward-Looking Risk Assessment** - Projections for {len(projections_2025)} active pitchers based on 2024 performance")
            
            # Team filter for projections
            if 'team' in projections_2025.columns:
                teams_2025_with_data = sorted([team for team in projections_2025['team'].unique() if pd.notna(team) and team != 'Unknown Team'])
                team_options_2025 = ["All Teams"] + teams_2025_with_data
                
                # Add 'Traded' option if there are players without team assignments
                traded_2025_count = len(projections_2025[projections_2025['team'] == 'Unknown Team'])
                if traded_2025_count > 0:
                    team_options_2025.append("Traded")
                
                selected_team_2025 = st.selectbox("⚾ Select Team (2025)", team_options_2025, index=0, key="team_2025")
                
                if selected_team_2025 == "All Teams":
                    filtered_projections = projections_2025.copy()
                elif selected_team_2025 == "Traded":
                    filtered_projections = projections_2025[projections_2025['team'] == 'Unknown Team'].copy()
                    st.info(f"🔄 Showing {len(filtered_projections)} traded players for 2025 projections")
                else:
                    filtered_projections = projections_2025[projections_2025['team'] == selected_team_2025].copy()
                    if len(filtered_projections) == 0:
                        st.warning(f"No 2025 projections found for {selected_team_2025}")
            else:
                filtered_projections = projections_2025.copy()
                st.info("Team assignments not available for 2025 projections")
            
            # 2025 Risk Overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 2025 Risk Distribution")
                risk_2025_counts = filtered_projections['risk_category_2025'].value_counts()
                
                fig_pie_2025 = px.pie(
                    values=risk_2025_counts.values,
                    names=risk_2025_counts.index,
                    color=risk_2025_counts.index,
                    color_discrete_map={
                        'Low': '#2E8B57',
                        'Moderate': '#FFD700', 
                        'High': '#FF8C00',
                        'Very High': '#DC143C'
                    },
                    title="Projected Risk for 2025 Season"
                )
                st.plotly_chart(fig_pie_2025, use_container_width=True)
            
            with col2:
                st.subheader("📊 Risk Summary")
                for category in ['Low', 'Moderate', 'High', 'Very High']:
                    count = len(filtered_projections[filtered_projections['risk_category_2025'] == category])
                    pct = count / len(filtered_projections) * 100
                    color_map = {'Low': '🟢', 'Moderate': '🟡', 'High': '🟠', 'Very High': '🔴'}
                    st.write(f"{color_map[category]} **{category}**: {count} pitchers ({pct:.1f}%)")
                
                # High-risk count for 2025
                high_risk_2025 = len(filtered_projections[filtered_projections['alert_level_2025'] >= 2])
                st.metric("🚨 High-Risk Pitchers", high_risk_2025, 
                         delta=f"{high_risk_2025/len(filtered_projections)*100:.1f}% of roster")
            
            # High-risk pitchers table for 2025
            st.subheader("🚨 High-Risk Pitchers for 2025")
            high_risk_pitchers_2025 = filtered_projections[filtered_projections['alert_level_2025'] >= 2].copy()
            
            if len(high_risk_pitchers_2025) > 0:
                # Sort by risk score
                high_risk_pitchers_2025 = high_risk_pitchers_2025.sort_values('risk_score_2025', ascending=False)
                
                # Display table
                display_cols = ['player_name', 'risk_category_2025', 'age', 'era_2024', 'g_2024', 'ip_2024']
                if 'team' in high_risk_pitchers_2025.columns:
                    display_cols.insert(1, 'team')
                
                display_df_2025 = high_risk_pitchers_2025[display_cols].copy()
                
                # Rename columns for display
                column_names = {
                    'player_name': 'Pitcher',
                    'team': 'Team',
                    'risk_category_2025': 'Projected Risk',
                    'age': 'Age (2025)',
                    'era_2024': '2024 ERA',
                    'g_2024': '2024 Games',
                    'ip_2024': '2024 IP'
                }
                display_df_2025 = display_df_2025.rename(columns=column_names)
                
                st.dataframe(
                    display_df_2025,
                    use_container_width=True,
                    height=400
                )
                
                # Download 2025 projections
                csv_2025 = filtered_projections.to_csv(index=False)
                st.download_button(
                    label=f"💾 Download 2025 Projections CSV",
                    data=csv_2025,
                    file_name=f"pitcher_projections_2025_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("🎉 No high-risk pitchers projected for this selection in 2025!")
            
            # Validation info
            st.subheader("🎯 Model Validation")
            try:
                import json
                with open('data/processed/validation_summary.json', 'r') as f:
                    validation_summary = json.load(f)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("2024 Validation C-index", f"{validation_summary['c_index_2024']:.3f}")
                with col2:
                    st.metric("Training Period", validation_summary['training_period'])
                with col3:
                    st.metric("Holdout Sample", f"{validation_summary['holdout_samples']} pitchers")
                
                st.info(f"📅 **Validation Summary**: External validation on 2024 data achieved C-index of {validation_summary['c_index_2024']:.3f} with monotonic risk gradient. Projections use 2024 statistics to predict 2025 injury risk.")
                
            except FileNotFoundError:
                st.warning("⚠️ Validation summary not found.")
        else:
            st.warning("🚨 2025 projections data not available. Please run the validation analysis to generate projections.")
            st.info("🔧 To generate 2025 projections, run: `python validate_2024_holdout.py`")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏥 MLB Pitcher Injury Risk Dashboard**")
    st.sidebar.markdown("*Powered by Bayesian Survival Analysis*")
    st.sidebar.markdown("*C-index: 0.607 (Good Discrimination)*")
    st.sidebar.markdown("*2024 Validation: 0.592 C-index*")

if __name__ == "__main__":
    main()