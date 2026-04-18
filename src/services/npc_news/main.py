import os
import argparse
import requests
from pathlib import Path
from typing import List
from playwright.sync_api import sync_playwright

from src.services.npc_news.db_manager import DatabaseManager
from src.etl.loading.drive_loading import upload_to_drive

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "STORAGE_BASE": "data/images",
    "DB_FILE": "data/npc_data.db",
    "GDRIVE_REMOTE": "gdrive:Bodybuilding_Dataset" 
}

class NPCNewsScraper:
    def __init__(self, years: List[int]):
        self.years = years
        self.db = DatabaseManager(CONFIG["DB_FILE"])
        self.img_session = requests.Session()
        self.img_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        })

    def _sanitize(self, name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def _parse_athlete_name(self, raw_text: str):
        parts = raw_text.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return int(parts[0]), parts[1].strip()
        return None, raw_text.strip()

    def download_image(self, url: str, path: Path):
        if path.exists(): return True
        try:
            res = self.img_session.get(url, timeout=15)
            if res.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(res.content)
                return True
        except: pass
        return False

    def run(self):
        with sync_playwright() as p:
            print("Launching browser with stealth settings...")
            browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context(viewport={'width': 1280, 'height': 800})
            page = context.new_page()
            
            print("Warming up session...")
            page.goto("https://contests.npcnewsonline.com/", wait_until="domcontentloaded", timeout=60000)

            for year in self.years:
                year_str = str(year)
                print(f"\n--- Processing Year: {year_str} ---")
                
                year_url = f"{CONFIG['BASE_URL']}{year}/"
                page.goto(year_url, wait_until="domcontentloaded", timeout=60000)
                
                contest_data = page.locator("a[href*='/contests/20']").evaluate_all("""
                    elements => elements.map(el => ({
                        href: el.getAttribute('href') || '',
                        name: el.innerText.trim()
                    }))
                """)
                
                unique_targets = []
                seen = set()
                for data in contest_data:
                    if data["name"] and f"/{year}/" in data["href"] and data["href"] not in seen:
                        seen.add(data["href"])
                        unique_targets.append(data)

                print(f"Found {len(unique_targets)} contests.")

                for contest in unique_targets:
                    c_name = contest["name"]
                    print(f"  Contest: {c_name}")

                    page.goto(year_url, wait_until="domcontentloaded", timeout=60000)
                    link_to_click = page.locator(f"a[href='{contest['href']}']").first
                    
                    try:
                        with page.expect_navigation(wait_until="domcontentloaded", timeout=15000):
                            link_to_click.click()
                    except:
                        page.goto(contest["href"], wait_until="domcontentloaded", timeout=15000)

                    c_path = page.url.replace("https://contests.npcnewsonline.com", "").rstrip("/")
                    links_data = page.locator("a").evaluate_all("""
                        elements => elements.map(el => ({
                            href: el.getAttribute('href') || '',
                            text: el.innerText.trim()
                        }))
                    """)
                    
                    ath_galleries = []
                    for data in links_data:
                        raw_href, text = data["href"], data["text"]
                        if not raw_href or not text or len(text) < 3: continue
                        
                        href_relative = raw_href.replace("https://contests.npcnewsonline.com", "")
                        
                        if c_path in href_relative and href_relative.rstrip("/") != c_path:
                            if text.lower() not in ["home", "back", "contests", "divisions"]:
                                
                                # --- Extract Division from URL! ---
                                sub_path = href_relative.replace(c_path, "").strip("/")
                                parts = sub_path.split("/")
                                division = parts[0] if len(parts) >= 2 else "overall"

                                ath_galleries.append({
                                    "raw_text": text, 
                                    "url": raw_href,
                                    "division": division
                                })

                    for ath in ath_galleries:
                        placing, ath_name = self._parse_athlete_name(ath["raw_text"])
                        division = ath["division"]
                        
                        # Check DB with Division included!
                        if self.db.is_athlete_processed(year, c_name, division, ath_name):
                            print(f"      Skipping {ath_name} [{division}] (Already in DB)")
                            continue

                        ath_url = ath["url"]
                        if not ath_url.startswith("http"):
                            ath_url = f"https://contests.npcnewsonline.com{ath_url}"
                        
                        target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str
                        
                        page.goto(ath_url, wait_until="domcontentloaded", timeout=60000)
                        
                        viewer_links = page.locator("a[href*='images.php']").evaluate_all("""
                            elements => elements.map(el => el.getAttribute('href'))
                        """)
                        
                        unique_viewers = list(set([href if href.startswith("http") else f"https://contests.npcnewsonline.com/{href.lstrip('/')}" for href in viewer_links if href]))

                        successful_images = 0
                        for idx, viewer_url in enumerate(unique_viewers):
                            try:
                                page.goto(viewer_url, wait_until="domcontentloaded", timeout=10000)
                                high_res_src = page.locator("img").evaluate_all("""
                                    elements => {
                                        for(let img of elements) {
                                            let src = img.getAttribute('src') || '';
                                            if(src.includes('/images/contests/') && !src.includes('thumb')) return src;
                                        }
                                        return null;
                                    }
                                """)
                                
                                if high_res_src:
                                    if not high_res_src.startswith("http"): 
                                        high_res_src = f"https://contests.npcnewsonline.com{high_res_src}"
                                    
                                    # FLAT FILE NAMING: 2011_Mr_Olympia_figure_Nicole_Wilkins_1.jpg
                                    clean_contest = self._sanitize(c_name)
                                    clean_div = self._sanitize(division)
                                    clean_ath = self._sanitize(ath_name)
                                    filename = f"{year_str}_{clean_contest}_{clean_div}_{clean_ath}_{idx+1}.jpg"
                                    local_path = target_dir / filename
                                    
                                    if self.download_image(high_res_src, local_path):
                                        # Save to Database! (local_path removed)
                                        self.db.insert_record(year, c_name, division, placing, ath_name, filename)
                                        successful_images += 1
                            except Exception:
                                pass 
                        
                        if successful_images > 0:
                            print(f"      Athlete: {ath_name} [{division}] -> Downloaded {successful_images} images")

            browser.close()
            self.db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Specific year to scrape (e.g., 2011)")
    parser.add_argument("--upload", action="store_true", help="Upload to Google Drive when finished")
    args = parser.parse_args()
    
    scraper = NPCNewsScraper([args.year] if args.year else list(range(2011, 2027)))
    scraper.run()
    
    if args.upload:
        upload_to_drive(CONFIG["STORAGE_BASE"], CONFIG["GDRIVE_REMOTE"])
        upload_to_drive(CONFIG["DB_FILE"], CONFIG["GDRIVE_REMOTE"])