import os
import json
import argparse
import requests
from pathlib import Path
from typing import List
from playwright.sync_api import sync_playwright

from src.etl.loading.drive_loading import upload_to_drive

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "STORAGE_BASE": "data/images",
    "METADATA_FILE": "data/images/metadata.json",
    "GDRIVE_REMOTE": "gdrive:Bodybuilding_Dataset" 
}

class NPCNewsScraper:
    def __init__(self, years: List[int]):
        self.years = years
        self.metadata = {}
        Path(CONFIG["STORAGE_BASE"]).mkdir(parents=True, exist_ok=True)
        self.img_session = requests.Session()
        self.img_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        })

    def _sanitize(self, name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def download_image(self, url: str, path: Path):
        """Downloads image and ONLY creates directories if the download is successful."""
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
            print("Launching browser...")
            browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context(viewport={'width': 1280, 'height': 800})
            page = context.new_page()
            
            print("Warming up session...")
            page.goto("https://contests.npcnewsonline.com/", wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            for year in self.years:
                year_str = str(year)
                print(f"\n--- Processing Year: {year_str} ---")
                self.metadata[year_str] = {}
                
                year_url = f"{CONFIG['BASE_URL']}{year}/"
                page.goto(year_url, wait_until="domcontentloaded")
                
                contest_data = page.locator("a[href*='/contests/20']").evaluate_all("""
                    elements => elements.map(el => ({
                        href: el.getAttribute('href') || '',
                        name: el.innerText.trim()
                    }))
                """)
                
                targets = []
                for data in contest_data:
                    name = data["name"]
                    href = data["href"]
                    if name and href and f"/{year}/" in href:
                        targets.append({"name": name, "href": href})
                
                unique_targets = []
                seen = set()
                for t in targets:
                    if t["href"] not in seen:
                        seen.add(t["href"])
                        unique_targets.append(t)

                print(f"Found {len(unique_targets)} contests. Starting stateful navigation...")

                for contest in unique_targets:
                    c_name = contest["name"]
                    print(f"  Contest: {c_name}")
                    self.metadata[year_str][c_name] = {}

                    page.goto(year_url, wait_until="domcontentloaded")
                    link_to_click = page.locator(f"a[href='{contest['href']}']").first
                    
                    try:
                        with page.expect_navigation(wait_until="domcontentloaded", timeout=15000):
                            link_to_click.click()
                        page.wait_for_timeout(2000)
                    except Exception as e:
                        print(f"    Click navigation failed for {c_name}. Using direct goto...")
                        page.goto(contest["href"], wait_until="domcontentloaded", timeout=15000)
                        page.wait_for_timeout(2000)

                    c_path = page.url.replace("https://contests.npcnewsonline.com", "").rstrip("/")
                    links_data = page.locator("a").evaluate_all("""
                        elements => elements.map(el => ({
                            href: el.getAttribute('href') || '',
                            text: el.innerText.trim()
                        }))
                    """)
                    
                    ath_galleries = []
                    for data in links_data:
                        href = data["href"]
                        text = data["text"]
                        
                        if not href or not text or len(text) < 3: continue
                        
                        if c_path in href and href.rstrip("/") != c_path:
                            if text.lower() not in ["home", "back", "contests", "divisions"]:
                                ath_galleries.append({"name": text, "url": href})

                    if not ath_galleries:
                        print(f"    Error: No athletes discovered for {c_name}.")
                        continue

                    print(f"    Found {len(ath_galleries)} athletes to process...")

                    for ath in ath_galleries:
                        ath_name = ath["name"]
                        ath_url = ath["url"]
                        if not ath_url.startswith("http"):
                            ath_url = f"https://contests.npcnewsonline.com{ath_url}"
                        
                        target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str / self._sanitize(c_name) / "General" / self._sanitize(ath_name)
                        
                        page.goto(ath_url, wait_until="domcontentloaded")
                        page.wait_for_timeout(1000)
                        
                        viewer_links = page.locator("a[href*='images.php']").evaluate_all("""
                            elements => elements.map(el => el.getAttribute('href'))
                        """)
                        
                        unique_viewers = []
                        for href in viewer_links:
                            if href:
                                full_url = href if href.startswith("http") else f"https://contests.npcnewsonline.com/{href.lstrip('/')}"
                                if full_url not in unique_viewers:
                                    unique_viewers.append(full_url)

                        successful_images = 0
                        for idx, viewer_url in enumerate(unique_viewers):
                            try:
                                page.goto(viewer_url, wait_until="domcontentloaded", timeout=10000)
                                
                                high_res_src = page.locator("img").evaluate_all("""
                                    elements => {
                                        for(let img of elements) {
                                            let src = img.getAttribute('src') || '';
                                            if(src.includes('/images/contests/') && !src.includes('thumb')) {
                                                return src;
                                            }
                                        }
                                        return null;
                                    }
                                """)
                                
                                if high_res_src:
                                    if not high_res_src.startswith("http"): 
                                        high_res_src = f"https://contests.npcnewsonline.com{high_res_src}"
                                    
                                    filename = f"image_{idx+1}.jpg"
                                    
                                    if self.download_image(high_res_src, target_dir / filename):
                                        self.metadata[year_str][c_name].setdefault("General", {}).setdefault(ath_name, []).append(filename)
                                        successful_images += 1
                            except Exception as e:
                                pass 
                        
                        if successful_images > 0:
                            print(f"      Athlete: {ath_name} -> Done: {successful_images} images")

                self._save_metadata()
            browser.close()

    def _save_metadata(self):
        with open(CONFIG["METADATA_FILE"], "w") as f:
            json.dump(self.metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Specific year to scrape (e.g., 2011)")
    parser.add_argument("--upload", action="store_true", help="Upload the dataset to Google Drive when finished")
    args = parser.parse_args()
    
    scraper = NPCNewsScraper([args.year] if args.year else list(range(2011, 2027)))
    scraper.run()
    
    if args.upload:
        upload_to_drive(CONFIG["STORAGE_BASE"], CONFIG["GDRIVE_REMOTE"])