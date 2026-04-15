import os
import json
import time
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "SELECTORS": {
        "CONTEST_LINKS": ".contest-listing a",                      # Links to individual contests from the year page
        "DIVISION_LINKS": ".contest-divisions a",                   # Links to divisions within a contest page
        "ATHLETE_CONTAINER": ".competitor-card",                    # Container for each athlete entry
        "ATHLETE_NAME": ".competitor-name",                         # Element containing athlete's name
        "ATHLETE_IMAGE": ".competitor-image img",                  # Image element for the athlete
    },
    "STORAGE_BASE": "data/images",
    "METADATA_FILE": "data/images/metadata.json",
    "REQUEST_DELAY": 1.5,
}

class NPCNewsScraper:
    def __init__(self, years: List[int]):
        self.years = years
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": CONFIG["USER_AGENT"]})
        self.metadata = {}
        
        # Ensure base directories exist
        Path(CONFIG["STORAGE_BASE"]).mkdir(parents=True, exist_ok=True)

    def _get_soup(self, url: str) -> BeautifulSoup:
        """Helper to fetch and parse HTML with polite delays."""
        print(f"Fetching: {url}")
        try:
            time.sleep(CONFIG["REQUEST_DELAY"])
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _sanitize(self, name: str) -> str:
        """Sanitizes strings for filesystem-safe directory/file names."""
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def download_image(self, url: str, path: Path):
        """Downloads image if not already present."""
        if path.exists():
            return True
        try:
            img_data = self.session.get(url, timeout=10).content
            with open(path, "wb") as f:
                f.write(img_data)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def run(self):
        for year in self.years:
            year_str = str(year)
            print(f"\n--- Processing Year: {year_str} ---")
            self.metadata[year_str] = {}
            
            year_url = f"{CONFIG['BASE_URL']}{year}/"
            soup = self._get_soup(year_url)
            if not soup: continue

            contests = soup.select(CONFIG["SELECTORS"]["CONTEST_LINKS"])
            for contest in contests:
                contest_name = contest.get_text(strip=True)
                contest_url = contest.get("href")
                if not contest_url.startswith("http"):
                    contest_url = f"https://contests.npcnewsonline.com{contest_url}"
                
                s_contest = self._sanitize(contest_name)
                self.metadata[year_str][contest_name] = {}
                print(f"  Contest: {contest_name}")

                c_soup = self._get_soup(contest_url)
                if not c_soup: continue

                divisions = c_soup.select(CONFIG["SELECTORS"]["DIVISION_LINKS"])
                for division in divisions:
                    div_name = division.get_text(strip=True)
                    div_url = division.get("href")
                    if not div_url.startswith("http"):
                        div_url = f"https://contests.npcnewsonline.com{div_url}"
                    
                    s_div = self._sanitize(div_name)
                    self.metadata[year_str][contest_name][div_name] = {}
                    
                    # Create directory: data/images/{year}/{contest}/{division}/
                    target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str / s_contest / s_div
                    target_dir.mkdir(parents=True, exist_ok=True)

                    d_soup = self._get_soup(div_url)
                    if not d_soup: continue

                    athletes = d_soup.select(CONFIG["SELECTORS"]["ATHLETE_CONTAINER"])
                    for athlete in athletes:
                        name_el = athlete.select_one(CONFIG["SELECTORS"]["ATHLETE_NAME"])
                        img_el = athlete.select_one(CONFIG["SELECTORS"]["ATHLETE_IMAGE"])
                        
                        if name_el and img_el:
                            ath_name = name_el.get_text(strip=True)
                            img_url = img_el.get("src") or img_el.get("data-src")
                            
                            if not img_url: continue
                            if not img_url.startswith("http"):
                                img_url = f"https://contests.npcnewsonline.com{img_url}"

                            filename = f"{self._sanitize(ath_name)}.jpg"
                            save_path = target_dir / filename
                            
                            if self.download_image(img_url, save_path):
                                self.metadata[year_str][contest_name][div_name][ath_name] = filename
            
            # Save metadata progress after each year
            self._save_metadata()

    def _save_metadata(self):
        with open(CONFIG["METADATA_FILE"], "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Metadata updated: {CONFIG['METADATA_FILE']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPC News Online Scraper")
    parser.add_argument("--year", type=int, help="Run for a specific year (test mode)")
    args = parser.parse_args()

    if args.year:
        years_to_run = [args.year]
    else:
        years_to_run = list(range(2011, 2027))

    scraper = NPCNewsScraper(years_to_run)
    try:
        scraper.run()
    except KeyboardInterrupt:
        print("\nStopping... Saving current progress.")
        scraper._save_metadata()
