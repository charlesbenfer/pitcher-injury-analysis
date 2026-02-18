"""
Launch script for the MLB Pitcher Risk Dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚀 LAUNCHING MLB PITCHER RISK DASHBOARD")
    print("=" * 45)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed")
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly found")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed")
    
    # Launch the dashboard
    print(f"\n🎯 Starting dashboard...")
    print(f"📂 Working directory: {os.getcwd()}")
    
    app_path = "app/pitcher_risk_app.py"
    if not os.path.exists(app_path):
        print(f"❌ Dashboard app not found at {app_path}")
        print("Please ensure you're running this from the project root directory")
        return
    
    print(f"🌐 Dashboard will open in your browser at: http://localhost:8501")
    print(f"⚾ Welcome to the MLB Pitcher Injury Risk Dashboard!")
    print(f"\n" + "="*50)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.address", "localhost",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()