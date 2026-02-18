#!/bin/bash

echo "Setting up Pitcher Injury Analysis Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create .env file for environment variables
echo "Creating .env file..."
cat > .env << EOL
# MLB Stats API credentials (if needed)
MLB_API_KEY=

# Database connection (if using)
DATABASE_URL=

# Other API keys
FANGRAPHS_API_KEY=
EOL

# Create gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch reports/figures/.gitkeep
touch tests/.gitkeep

echo "Setup complete! Activate the environment with: source venv/bin/activate"