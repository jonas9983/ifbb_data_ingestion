import os
import argparse
import requests
import re
import time
import random
import threading
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

from src.etl.loading.drive_loading import upload_to_drive
from src.etl.extraction.drive_extraction import download_from_drive
from src.etl.loading.db_loading import DatabaseManager

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "STORAGE_BASE": "data/npc_news",
    "GDRIVE_REMOTE": "gdrive:personal/Bodybuilding_Dataset",
    "DISK_LIMIT_GB": 20,
    "MAX_WORKERS": 5,
    "DB_NAME": "npc_data_2013_fw.db"
}

class NPCNewsScraper:
    def __init__(self, years: List[int]):
        self.years = years
        
        # 1. Try to download the latest database from Drive before starting
        db_path = f"data/{CONFIG['DB_NAME']}"
        remote_db_path = f"{CONFIG['GDRIVE_REMOTE']}/{CONFIG['DB_NAME']}"
        print(f"[Startup] Checking for remote database at {remote_db_path}...")
        download_from_drive(remote_db_path, db_path)

        # 2. Initialize Database with the (potentially updated) local file
        self.db = DatabaseManager(db_path)
        
        self.img_session = requests.Session()
        self.img_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        })

        # 3. Async Upload State
        self.upload_executor = ThreadPoolExecutor(max_workers=1)
        self._upload_queued = False
        self._upload_lock = threading.Lock()

        # 4. Storage Tracking
        self.total_size_bytes = self._calculate_initial_size()

    def _calculate_initial_size(self) -> int:
        """Calculates initial size of STORAGE_BASE in bytes."""
        root_directory = Path(CONFIG["STORAGE_BASE"])
        if not root_directory.exists(): return 0
        return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())

    def _sanitize(self, name: str) -> str:
        """Cleans strings for filesystem safety."""
        return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    def _parse_athlete_name(self, raw_text: str):
        """Extracts placing (int) and name from '1 PHIL HEATH'."""
        parts = raw_text.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return int(parts[0]), parts[1].strip()
        return None, raw_text.strip()

    def _get_storage_size_gb(self) -> float:
        """Returns total size of STORAGE_BASE in GB using tracked value."""
        return self.total_size_bytes / (1024**3)

    def _upload_and_cleanup(self, year: int = None, backup_db: bool = False):
        """Uploads STORAGE_BASE and optionally backs up Database to Drive."""
        print(f"\n[Maintenance] Triggering upload...")
        
        # 1. Upload/Move the images
        if year:
            year_str = str(year)
            source_path = str(Path(CONFIG["STORAGE_BASE"]) / year_str)
            remote_path = f"{CONFIG['GDRIVE_REMOTE']}/{year_str}"
            if os.path.exists(source_path):
                print(f"Syncing year {year_str} specifically to improve speed...")
                upload_to_drive(source_path, remote_path, delete_after=True)
        else:
            upload_to_drive(CONFIG["STORAGE_BASE"], CONFIG["GDRIVE_REMOTE"], delete_after=True)
        
        # Recalculate size after move/cleanup
        self.total_size_bytes = self._calculate_initial_size()

        # 2. Backup the database (Only if requested or major sync)
        if backup_db:
            db_file = Path(f"data/{CONFIG['DB_NAME']}")
            if db_file.exists():
                print(f"Backing up database {db_file.name} to Drive...")
                upload_to_drive(str(db_file), CONFIG["GDRIVE_REMOTE"])

        # Ensure STORAGE_BASE exists for next batches
        Path(CONFIG["STORAGE_BASE"]).mkdir(parents=True, exist_ok=True)

    def _download_task(self, url: str, path: Path, year, contest, division, placing, athlete):
        """Pure binary download task for the thread pool."""
        size = self.download_image(url, path)
        if size > 0:
            with self._upload_lock:
                self.total_size_bytes += size
            self.db.insert_record(year, contest, division, placing, athlete, path.name, commit=False)
            return True
        return False

    def download_image(self, url: str, path: Path) -> int:
        """Downloads the raw binary image file and returns its size."""
        if path.exists(): return 0
        try:
            res = self.img_session.get(res.url if hasattr(res, 'url') else url, timeout=15)
            if res.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(res.content)
                return len(res.content)
        except: pass
        return 0

    def _async_upload_and_cleanup(self, year: int = None, backup_db: bool = False):
        """Queues an upload task if one isn't already queued."""
        with self._upload_lock:
            if self._upload_queued:
                return
            self._upload_queued = True
        
        def task_wrapper():
            try:
                self._upload_and_cleanup(year, backup_db=backup_db)
            finally:
                with self._upload_lock:
                    self._upload_queued = False
        
        self.upload_executor.submit(task_wrapper)

    def run(self):
        # Ensure storage base exists
        Path(CONFIG["STORAGE_BASE"]).mkdir(parents=True, exist_ok=True)
        
        try:
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
                    
                    contest_data = page.locator(".td-pb-span8.td-main-content a[href*='/contests/20']").evaluate_all("""
                        elements => elements.map(el => ({
                            href: el.getAttribute('href') || '',
                            name: el.innerText.trim()
                        }))
                    """)
                    
                    unique_targets = []
                    seen = set()
                    for data in contest_data:
                        href = data["href"].lower()
                        name_lower = data["name"].lower()
                        
                        # Extract the part of the URL after the year to avoid matching the domain (npcnewsonline.com)
                        path_part = href.split(f"/{year}/")[-1] if f"/{year}/" in href else href
                        
                        # Filter out non-IFBB organizations (NPC, NPC Worldwide, CPA)
                        exclude_patterns = ["npc", "npcw", "cpa", "npc_worldwide"]
                        
                        is_non_ifbb = any(p in path_part for p in exclude_patterns) or \
                                      (( "npc" in name_lower or "cpa" in name_lower ) and "ifbb" not in name_lower)

                        if is_non_ifbb:
                            continue

                        if data["name"] and f"/{year}/" in data["href"] and data["href"] not in seen:
                            seen.add(data["href"])
                            unique_targets.append(data)

                    print(f"Found {len(unique_targets)} contests.")

                    for contest in unique_targets:
                        c_name = contest["name"]
                        print(f"  Contest: {c_name}")

                        # Track new images saved in THIS contest
                        contest_new_images = 0

                        # --- PRE-FETCH PROCESSED ATHLETES ---
                        processed_set = self.db.get_processed_athletes_for_contest(year, c_name)

                        # Optimization: Navigate directly to the contest URL
                        page.goto(contest["href"], wait_until="domcontentloaded", timeout=30000)

                        c_path = page.url.replace("https://contests.npcnewsonline.com", "").rstrip("/")
                        links_data = page.evaluate("""() => {
                            let results = [];
                            let currentDiv = "Overall";
                            let elements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, .division, .title, a');
                            for(let el of elements) {
                                if(el.tagName.toUpperCase() === 'A') {
                                    let href = el.getAttribute('href') || '';
                                    let text = el.innerText.trim();
                                    if(href && text.length > 2) results.push({ href: href, text: text, division: currentDiv });
                                } else {
                                    let text = el.innerText.trim();
                                    if(text && text.length > 2 && text.length < 50) {
                                        if(!text.toLowerCase().includes("galleries") && !text.toLowerCase().includes("articles") && !text.toLowerCase().includes("years")) {
                                            currentDiv = text;
                                        }
                                    }
                                }
                            }
                            return results;
                        }""")
                        
                        ath_galleries = []
                        for data in links_data:
                            raw_href, text, div_name = data["href"], data["text"], data["division"]
                            href_rel = raw_href.replace("https://contests.npcnewsonline.com", "")
                            
                            if c_path in href_rel and href_rel.rstrip("/") != c_path:
                                if text.lower() not in ["home", "back", "contests", "divisions"]:
                                    ath_galleries.append({"raw_text": text, "url": raw_href, "division": div_name})

                        print(f"    Found {len(ath_galleries)} athletes/links. Processing...")

                        for idx_ath, ath in enumerate(ath_galleries):
                            placing, ath_name = self._parse_athlete_name(ath["raw_text"])
                            division = ath["division"]
                            
                            # --- IDEMPOTENCY CHECK (In Memory) ---
                            if (division, ath_name) in processed_set:
                                # Log progress without flooding
                                if (idx_ath + 1) % 100 == 0 or idx_ath == 0 or (idx_ath + 1) == len(ath_galleries):
                                    print(f"      [{idx_ath+1}/{len(ath_galleries)}] Skipping {ath_name} (Already in DB)")
                                continue

                            print(f"      [{idx_ath+1}/{len(ath_galleries)}] Processing Athlete: {ath_name}...")
                            ath_url = ath["url"]
                            if not ath_url.startswith("http"):
                                ath_url = f"https://contests.npcnewsonline.com{ath_url}"
                            
                            target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str
                            page.goto(ath_url, wait_until="domcontentloaded", timeout=60000)
                            
                            viewer_links = page.locator("a[href*='images.php']").evaluate_all("""
                                elements => elements.map(el => el.getAttribute('href'))
                            """)
                            
                            unique_viewers = list(set([href if href.startswith("http") else f"https://contests.npcnewsonline.com/{href.lstrip('/')}" for href in viewer_links if href]))

                            image_targets = []
                            for idx, v_url in enumerate(unique_viewers):
                                try:
                                    page.goto(v_url, wait_until="domcontentloaded", timeout=10000)
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
                                        
                                        c_clean = self._sanitize(c_name)
                                        d_clean = self._sanitize(division)
                                        a_clean = self._sanitize(ath_name)
                                        filename = f"{year_str}_{c_clean}_{d_clean}_{a_clean}_{idx+1}.jpg"
                                        image_targets.append((high_res_src, target_dir / filename))
                                except:
                                    continue

                            successful_images = 0
                            if image_targets:
                                with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
                                    futures = [executor.submit(self._download_task, url, path, year, c_name, division, placing, ath_name) for url, path in image_targets]
                                    for f in futures:
                                        if f.result(): successful_images += 1
                            
                            if successful_images > 0:
                                self.db.commit()
                                contest_new_images += successful_images
                                print(f"      Athlete: {ath_name} [{division}] -> {successful_images} images saved.")
                            else:
                                print(f"      Athlete: {ath_name} [{division}] -> No images found.")

                            if self._get_storage_size_gb() >= CONFIG["DISK_LIMIT_GB"]:
                                self._async_upload_and_cleanup(year, backup_db=True)

                        # Upload images for the year but don't force DB backup every contest unless disk is full
                        if contest_new_images > 0:
                            self._async_upload_and_cleanup(year, backup_db=False)

                browser.close()
        finally:
            print("\n[Shutdown] Performing final database backup and waiting for background uploads...")
            self._upload_and_cleanup(backup_db=True) # Final synchronous cleanup
            self.upload_executor.shutdown(wait=True)
            self.db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Specific year to scrape")
    args = parser.parse_args()

    scraper = NPCNewsScraper([args.year] if args.year else list(range(2013, 2027)))
    scraper.run()
