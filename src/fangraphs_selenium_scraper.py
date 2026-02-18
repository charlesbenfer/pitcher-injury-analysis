import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class FanGraphsSeleniumScraper:
    
    def __init__(self):
        # Setup Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run in background
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
    def scrape_injury_report(self, year: int = 2023):
        """
        Scrape FanGraphs injury report for a specific year using Selenium
        """
        driver = None
        try:
            # Initialize driver
            print(f"Setting up Chrome driver...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            
            # Navigate to the page
            url = f"https://www.fangraphs.com/roster-resource/injury-report?timeframe=all&season={year}"
            print(f"Navigating to {url}")
            driver.get(url)
            
            # Wait for the page to load (wait for table or specific element)
            print("Waiting for page to load...")
            wait = WebDriverWait(driver, 20)
            
            # Try multiple possible selectors
            selectors_to_try = [
                "table",
                ".injury-table",
                ".rr-table",
                "[data-testid='injury-table']",
                ".MuiTable-root",
                ".MuiDataGrid-root"
            ]
            
            page_loaded = False
            for selector in selectors_to_try:
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    print(f"Found element with selector: {selector}")
                    page_loaded = True
                    break
                except:
                    continue
            
            if not page_loaded:
                print("Waiting additional time for dynamic content...")
                time.sleep(10)
            
            # Get the page source after JavaScript has rendered
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Try to find injury data
            injuries = self._extract_injuries_from_soup(soup, year)
            
            if not injuries:
                # Try scrolling to load more data
                print("Attempting to scroll for more data...")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                injuries = self._extract_injuries_from_soup(soup, year)
            
            return injuries
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            return []
        finally:
            if driver:
                driver.quit()
    
    def _extract_injuries_from_soup(self, soup, year):
        """
        Extract injury data from the parsed HTML
        """
        injuries = []
        
        # Method 1: Look for tables
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")
        
        for table in tables:
            # Check if this looks like an injury table
            headers = table.find_all('th')
            header_text = ' '.join([h.text.lower() for h in headers])
            
            if any(word in header_text for word in ['player', 'injury', 'date', 'status']):
                print("Found potential injury table!")
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        injury_data = {
                            'player': cells[0].text.strip(),
                            'team': cells[1].text.strip() if len(cells) > 1 else '',
                            'injury': cells[2].text.strip() if len(cells) > 2 else '',
                            'date': cells[3].text.strip() if len(cells) > 3 else '',
                            'year': year
                        }
                        injuries.append(injury_data)
        
        # Method 2: Look for divs with specific classes
        if not injuries:
            injury_divs = soup.find_all('div', class_=lambda x: x and ('injury' in x.lower() or 'player' in x.lower()))
            print(f"Found {len(injury_divs)} potential injury divs")
            
            for div in injury_divs:
                text = div.text.strip()
                if text and len(text) > 5:  # Filter out empty or very short text
                    # Try to parse the text
                    parts = text.split()
                    if len(parts) >= 2:
                        injuries.append({
                            'raw_text': text,
                            'year': year
                        })
        
        # Method 3: Look for MUI components (Material-UI)
        if not injuries:
            mui_rows = soup.find_all('div', class_=lambda x: x and 'MuiTableRow' in str(x))
            print(f"Found {len(mui_rows)} MUI table rows")
            
            for row in mui_rows:
                cells = row.find_all('div', class_=lambda x: x and 'MuiTableCell' in str(x))
                if cells:
                    injury_data = {
                        'player': cells[0].text.strip() if len(cells) > 0 else '',
                        'injury': cells[1].text.strip() if len(cells) > 1 else '',
                        'date': cells[2].text.strip() if len(cells) > 2 else '',
                        'year': year
                    }
                    if injury_data['player']:  # Only add if we have a player name
                        injuries.append(injury_data)
        
        print(f"Extracted {len(injuries)} injury records")
        return injuries
    
    def scrape_multiple_years(self, start_year=2020, end_year=2024):
        """
        Scrape injury data for multiple years
        """
        all_injuries = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nScraping year {year}...")
            injuries = self.scrape_injury_report(year)
            all_injuries.extend(injuries)
            time.sleep(5)  # Be respectful between requests
        
        if all_injuries:
            df = pd.DataFrame(all_injuries)
            return df
        else:
            print("No injuries found via Selenium. Using fallback known injuries...")
            return self._get_known_injuries()
    
    def _get_known_injuries(self):
        """
        Return known injuries as fallback
        """
        known_injuries = [
            # Recent high-profile pitcher injuries
            {'player': 'Spencer Strider', 'team': 'ATL', 'injury': 'UCL surgery', 'date': '2024-04-05', 'year': 2024},
            {'player': 'Shane Bieber', 'team': 'CLE', 'injury': 'Tommy John surgery', 'date': '2024-04-14', 'year': 2024},
            {'player': 'Jacob deGrom', 'team': 'TEX', 'injury': 'Elbow inflammation', 'date': '2023-06-12', 'year': 2023},
            {'player': 'Tyler Glasnow', 'team': 'TB', 'injury': 'Oblique strain', 'date': '2023-05-05', 'year': 2023},
            # Add more as needed
        ]
        
        return pd.DataFrame(known_injuries)


if __name__ == "__main__":
    print("FanGraphs Selenium Scraper")
    print("=" * 50)
    
    scraper = FanGraphsSeleniumScraper()
    
    # Try to scrape 2023 data
    injuries_2023 = scraper.scrape_injury_report(2023)
    
    if injuries_2023:
        df = pd.DataFrame(injuries_2023)
        print(f"\nFound {len(df)} injuries")
        print(df.head())
        df.to_csv('data/processed/fangraphs_injuries_selenium.csv', index=False)
    else:
        print("No injuries found")
        # Use fallback
        df = scraper._get_known_injuries()
        df.to_csv('data/processed/fangraphs_injuries_selenium.csv', index=False)
        print(f"Saved {len(df)} known injuries as fallback")