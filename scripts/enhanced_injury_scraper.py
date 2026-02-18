"""
Enhanced MLB Injury Data Scraper for Pitcher Analysis
Based on methods from https://github.com/ssharpe42/mlb-injury

This script combines multiple data sources to create a comprehensive 
pitcher injury dataset for 2018-2024 seasons.
"""

import logging
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBInjuryScraperEnhanced:
    """Enhanced MLB injury scraper with multiple data sources"""
    
    def __init__(self, data_dir="../data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        
    def scrape_team_info(self, year):
        """Scrape team information for a given year"""
        teams_url = f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={year}"
        
        try:
            response = requests.get(teams_url)
            response.raise_for_status()
            mlb_teams = response.json()["teams"]
            
            teams_yr = pd.DataFrame([
                {
                    "id": team["id"],
                    "teamName": team["teamName"], 
                    "abbreviation": team["abbreviation"]
                }
                for team in mlb_teams
            ]).rename(columns={
                "abbreviation": "team",
                "teamName": "team_name", 
                "id": "team_id"
            })
            teams_yr["year"] = year
            
            logger.info(f"Successfully scraped team info for {year}")
            return teams_yr
            
        except requests.RequestException as e:
            logger.error(f"Failed to scrape team info for {year}: {e}")
            return pd.DataFrame()
    
    def scrape_il_data(self, start_year, end_year):
        """Scrape IL (Injured List) data from MLB Stats API"""
        url = "https://statsapi.mlb.com/api/v1/transactions?startDate={start}&endDate={end}"
        
        all_status_changes = []
        teams_list = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Scraping IL data for {year}")
            
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            try:
                response = requests.get(url.format(start=start_date, end=end_date))
                response.raise_for_status()
                results = response.json()["transactions"]
                
                # Filter for status changes
                status_changes = [x for x in results if x["typeCode"] == "SC"]
                
                for trxn in status_changes:
                    if "person" in trxn and "description" in trxn:
                        temp = {
                            k: v for k, v in trxn.items()
                            if k in ["id", "date", "effectiveDate", "resolutionDate", "description"]
                        }
                        temp["player_id"] = trxn["person"]["id"]
                        temp["team_id"] = trxn["toTeam"]["id"]
                        all_status_changes.append(temp)
                
                # Get team info
                teams_list.append(self.scrape_team_info(year))
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except requests.RequestException as e:
                logger.error(f"Failed to scrape IL data for {year}: {e}")
                continue
        
        if not all_status_changes:
            logger.warning("No status changes found")
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine data
        teams = pd.concat(teams_list, ignore_index=True)
        status_changes = pd.DataFrame(all_status_changes)
        
        # Process dates with error handling
        date_cols = ["date", "effectiveDate", "resolutionDate"]
        for col in date_cols:
            if col in status_changes.columns:
                status_changes[col] = pd.to_datetime(status_changes[col], errors='coerce')
        
        status_changes["year"] = status_changes.date.dt.year
        
        # Merge with team info
        status_changes = status_changes.merge(teams, on=["team_id", "year"], how="inner")
        
        logger.info(f"Scraped {len(status_changes)} status changes")
        return status_changes, teams
    
    def scrape_dtd_data_prosports(self, start_year, end_year):
        """Scrape Day-to-Day data from Pro Sports Transactions"""
        base_url = "https://www.prosportstransactions.com/baseball/Search/SearchResults.php"
        
        all_dtd_data = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Scraping DTD data for {year} from Pro Sports")
            
            params = {
                'Player': '',
                'Team': '',
                'BeginDate': f'1/1/{year}',
                'EndDate': f'12/31/{year}',
                'PlayerMovementChkBx': 'yes',
                'Submit': 'Search',
                'start': 0
            }
            
            start_idx = 0
            while True:
                try:
                    params['start'] = start_idx
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()
                    
                    # Parse HTML tables
                    tables = pd.read_html(response.text)
                    
                    if not tables or len(tables[0]) == 0:
                        break
                    
                    df = tables[0]
                    df['year'] = year
                    all_dtd_data.append(df)
                    
                    # Check if we need more pages
                    if len(df) < 25:  # Assuming 25 results per page
                        break
                    
                    start_idx += 25
                    time.sleep(self.request_delay)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Failed to scrape DTD data for {year}, start={start_idx}: {e}")
                    break
        
        if all_dtd_data:
            dtd_combined = pd.concat(all_dtd_data, ignore_index=True)
            logger.info(f"Scraped {len(dtd_combined)} DTD records")
            return dtd_combined
        else:
            logger.warning("No DTD data found")
            return pd.DataFrame()
    
    def get_pitcher_data(self, start_year, end_year):
        """Get pitcher-specific data from MLB Stats API"""
        pitcher_data = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Getting pitcher data for {year}")
            
            # Get all players for the season
            players_url = f"https://statsapi.mlb.com/api/v1/sports/1/players?season={year}"
            
            try:
                response = requests.get(players_url)
                response.raise_for_status()
                players = response.json()["people"]
                
                # Filter for pitchers
                pitchers = [
                    p for p in players 
                    if p.get("primaryPosition", {}).get("code") == "1"
                ]
                
                for pitcher in pitchers:
                    pitcher_info = {
                        "player_id": pitcher["id"],
                        "full_name": pitcher["fullName"],
                        "first_name": pitcher.get("firstName", ""),
                        "last_name": pitcher.get("lastName", ""),
                        "birth_date": pitcher.get("birthDate"),
                        "height": pitcher.get("height"),
                        "weight": pitcher.get("weight"),
                        "throws": pitcher.get("pitchHand", {}).get("code"),
                        "year": year
                    }
                    pitcher_data.append(pitcher_info)
                
                time.sleep(self.request_delay)
                
            except requests.RequestException as e:
                logger.error(f"Failed to get pitcher data for {year}: {e}")
                continue
        
        if pitcher_data:
            pitchers_df = pd.DataFrame(pitcher_data)
            logger.info(f"Retrieved {len(pitchers_df)} pitcher records")
            return pitchers_df
        else:
            return pd.DataFrame()
    
    def filter_pitcher_injuries(self, injuries_df, pitchers_df):
        """Filter injuries to only include pitchers"""
        pitcher_injuries = injuries_df.merge(
            pitchers_df[["player_id", "year"]], 
            on=["player_id", "year"],
            how="inner"
        )
        
        logger.info(f"Filtered to {len(pitcher_injuries)} pitcher injury records")
        return pitcher_injuries
    
    def process_injury_data(self, injuries_df):
        """Process and clean injury data"""
        logger.info("Processing injury data")
        
        # Extract IL days from description
        injuries_df["il_days"] = injuries_df["description"].str.extract(
            r"the (\d+)(?:\s|-)day", expand=False
        ).astype(float)
        
        # Identify injury types
        injuries_df["is_il"] = injuries_df["description"].str.contains(
            r"the (\d+)(?:\s|-)day", na=False
        )
        injuries_df["is_dtd"] = injuries_df["description"].str.contains(
            "day-to-day", case=False, na=False
        )
        injuries_df["is_activated"] = injuries_df["description"].str.contains(
            "activat", case=False, na=False
        )
        
        # Extract injury location/type from description
        injury_keywords = {
            "arm": ["arm", "forearm"],
            "shoulder": ["shoulder"],
            "elbow": ["elbow", "ucl"],
            "back": ["back", "spine"],
            "leg": ["leg", "hamstring", "quad", "calf"],
            "hand": ["hand", "finger", "wrist"],
            "head": ["head", "concussion"],
            "other": []
        }
        
        injuries_df["injury_location"] = "other"
        for location, keywords in injury_keywords.items():
            if keywords:  # Skip empty keyword lists
                pattern = "|".join(keywords)
                mask = injuries_df["description"].str.contains(
                    pattern, case=False, na=False
                )
                injuries_df.loc[mask, "injury_location"] = location
        
        return injuries_df
    
    def scrape_all_data(self, start_year=2018, end_year=2024):
        """Main method to scrape all injury data"""
        logger.info(f"Starting comprehensive injury data scrape for {start_year}-{end_year}")
        
        # 1. Scrape IL data from Stats API
        logger.info("Step 1: Scraping IL data from MLB Stats API")
        il_data, teams = self.scrape_il_data(start_year, end_year)
        
        # Save IL data
        il_file = self.data_dir / f"il_data_{start_year}_{end_year}.csv"
        il_data.to_csv(il_file, index=False)
        logger.info(f"Saved IL data to {il_file}")
        
        # 2. Scrape DTD data from Pro Sports
        logger.info("Step 2: Scraping DTD data from Pro Sports")
        dtd_data = self.scrape_dtd_data_prosports(start_year, end_year)
        
        if not dtd_data.empty:
            dtd_file = self.data_dir / f"dtd_data_{start_year}_{end_year}.csv"
            dtd_data.to_csv(dtd_file, index=False)
            logger.info(f"Saved DTD data to {dtd_file}")
        
        # 3. Get pitcher information
        logger.info("Step 3: Getting pitcher data")
        pitchers = self.get_pitcher_data(start_year, end_year)
        
        if not pitchers.empty:
            pitchers_file = self.data_dir / f"pitchers_{start_year}_{end_year}.csv"
            pitchers.to_csv(pitchers_file, index=False)
            logger.info(f"Saved pitcher data to {pitchers_file}")
        
        # 4. Combine and process all data
        logger.info("Step 4: Combining and processing data")
        
        if not il_data.empty:
            # Filter for pitcher injuries
            pitcher_injuries = self.filter_pitcher_injuries(il_data, pitchers)
            
            # Process injury data
            pitcher_injuries = self.process_injury_data(pitcher_injuries)
            
            # Save final dataset
            final_file = self.data_dir / f"pitcher_injuries_{start_year}_{end_year}.csv"
            pitcher_injuries.to_csv(final_file, index=False)
            logger.info(f"Saved final pitcher injury dataset to {final_file}")
            
            # Print summary statistics
            logger.info(f"""
            SCRAPING SUMMARY:
            ================
            Total pitcher injury records: {len(pitcher_injuries)}
            IL injuries: {pitcher_injuries['is_il'].sum()}
            DTD injuries: {pitcher_injuries['is_dtd'].sum()}
            Activations: {pitcher_injuries['is_activated'].sum()}
            Years covered: {start_year}-{end_year}
            Unique pitchers: {pitcher_injuries['player_id'].nunique()}
            
            Injury locations:
            {pitcher_injuries['injury_location'].value_counts().to_dict()}
            """)
            
            return pitcher_injuries
        
        else:
            logger.error("No IL data found - cannot proceed with analysis")
            return pd.DataFrame()


if __name__ == "__main__":
    # Create scraper instance
    scraper = MLBInjuryScraperEnhanced()
    
    # Scrape data for 2018-2024
    pitcher_injuries = scraper.scrape_all_data(start_year=2018, end_year=2024)
    
    if not pitcher_injuries.empty:
        print(f"\nScraping completed successfully!")
        print(f"Dataset shape: {pitcher_injuries.shape}")
        print(f"Data saved to: {scraper.data_dir}")
    else:
        print("Scraping failed - no data collected")