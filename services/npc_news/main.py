import os
import argparse
import requests
import re
import time
import random
import threading
import shutil
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

from ifbb_data_ingestion.etl.loading.drive_loading import upload_to_drive
from ifbb_data_ingestion.etl.extraction.drive_extraction import download_from_drive
from ifbb_data_ingestion.etl.loading.db_loading import DatabaseManager

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "STORAGE_BASE": "data/npc_news",
    "GDRIVE_REMOTE": "gdrive:personal/Bodybuilding_Dataset",
    "DISK_LIMIT_GB": 20,
    "MAX_WORKERS": 2,
    "DB_NAME": "npc_database.db"
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

    def _upload_and_cleanup_contest(self, year: int, contest_name: str, contest_dir: Path, backup_db: bool = False):
        """Zips the contest directory, uploads to Drive, and cleans up."""
        timestamp = time.strftime("%H:%M:%S")
        c_clean = self._sanitize(contest_name)
        zip_path = Path("data/upload_staging") / f"{year}_{c_clean}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{timestamp}] [Background Task] Zipping and syncing contest: {contest_name} ({year})...")
        
        # 1. Zip the contest directory
        if contest_dir.exists() and any(contest_dir.iterdir()):
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(contest_dir))
            
            # 2. Upload the zip to Drive into a year-specific folder
            remote_dest = f"{CONFIG['GDRIVE_REMOTE']}/{year}"
            print(f"  [{timestamp}] [Sync] Uploading {zip_path.name} to Drive...")
            upload_to_drive(str(zip_path), remote_dest, delete_after=True)
            
            # Clean up the local contest directory
            try:
                shutil.rmtree(contest_dir)
            except Exception as e:
                print(f"  [Error] Failed to remove local contest dir {contest_dir}: {e}")
            
            # Clean up the local zip if delete_after didn't get it (or if it's left behind)
            if zip_path.exists():
                try: zip_path.unlink()
                except: pass

        # 3. Backup the database (Only if requested or major sync)
        if backup_db:
            db_file = Path(f"data/{CONFIG['DB_NAME']}")
            if db_file.exists():
                print(f"  [{timestamp}] [Backup] Creating safe database snapshot...")
                backup_file = db_file.with_suffix(".db.tmp")
                try:
                    self.db.backup(str(backup_file))
                    print(f"  [{timestamp}] [Backup] Uploading database snapshot to Drive...")
                    # Upload with the original name to Drive
                    upload_to_drive(str(backup_file), f"{CONFIG['GDRIVE_REMOTE']}/{CONFIG['DB_NAME']}")
                except Exception as e:
                    print(f"  [Error] Failed to backup database: {e}")
                finally:
                    if backup_file.exists():
                        backup_file.unlink()

        with self._upload_lock:
            self.total_size_bytes = self._calculate_initial_size()
            
        print(f"[{time.strftime('%H:%M:%S')}] [Background Task] Sync for {contest_name} completed.\n")

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
        if path.exists():
            return path.stat().st_size
        try:
            res = self.img_session.get(url, timeout=15)
            if res.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(res.content)
                return len(res.content)
        except: pass
        return 0

    def _async_upload_and_cleanup_contest(self, year: int, contest_name: str, contest_dir: Path, backup_db: bool = False):
        """Queues an upload task for a contest."""
        def task_wrapper():
            self._upload_and_cleanup_contest(year, contest_name, contest_dir, backup_db=backup_db)
        
        self.upload_executor.submit(task_wrapper)

    def _find_high_res_src(self, v_url: str):
        """Finds high-res image source from a viewer page using fast requests."""
        try:
            res = self.img_session.get(v_url, timeout=10)
            if res.status_code == 200:
                # Optimized regex: look for ANY .jpg inside src that isn't a thumb
                # This is more robust for Awards pages
                patterns = [
                    r'src=["\']([^"\']+/images/contests/(?!.*thumb)[^"\']+\.jpg)["\']',
                    r'src=["\']([^"\']+/images/contests/[^"\']+\.jpg)["\']' # Fallback
                ]
                for p in patterns:
                    match = re.search(p, res.text, re.IGNORECASE)
                    if match:
                        src = match.group(1)
                        if "thumb" in src.lower() and len(patterns) > 1: continue
                        if not src.startswith("http"):
                            src = f"https://contests.npcnewsonline.com{src}"
                        return src
        except: pass
        return None

    def run(self):
        Path(CONFIG["STORAGE_BASE"]).mkdir(parents=True, exist_ok=True)
        
        try:
            with sync_playwright() as p:
                print("Launching browser with stealth settings...")
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                context = browser.new_context(viewport={'width': 1280, 'height': 800})
                page = context.new_page()
                
                print("Warming up session...")
                try:
                    page.goto("https://contests.npcnewsonline.com/", wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    print(f"  [Warning] Warmup navigation timeout: {e}. Proceeding anyway...")

                for year in self.years:
                    year_str = str(year)
                    print(f"\n--- Processing Year: {year_str} ---")
                    
                    year_url = f"{CONFIG['BASE_URL']}{year}/"
                    try:
                        page.goto(year_url, wait_until="domcontentloaded", timeout=60000)
                    except:
                        print(f"  [Error] Timeout loading year {year}. Skipping...")
                        continue
                    
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
                        path_part = href.split(f"/{year}/")[-1] if f"/{year}/" in href else href
                        exclude_patterns = ["npc", "npcw", "cpa", "npc_worldwide"]
                        is_non_ifbb = any(p in path_part for p in exclude_patterns) or \
                                      (( "npc" in name_lower or "cpa" in name_lower ) and "ifbb" not in name_lower)

                        if is_non_ifbb: continue

                        if data["name"] and f"/{year}/" in data["href"] and data["href"] not in seen:
                            seen.add(data["href"])
                            unique_targets.append(data)

                    print(f"Found {len(unique_targets)} contests.")

                    for contest in unique_targets:
                        c_name = contest["name"]
                        c_clean = self._sanitize(c_name)
                        target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str / c_clean

                        print(f"  Contest: {c_name}")
                        contest_new_images = 0
                        processed_set = self.db.get_processed_athletes_for_contest(year, c_name)

                        try:
                            page.goto(contest["href"], wait_until="domcontentloaded", timeout=30000)
                        except:
                            print(f"    [Error] Timeout loading contest {c_name}. Skipping...")
                            continue

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
                            
                            if (division, ath_name) in processed_set:
                                if (idx_ath + 1) % 100 == 0 or idx_ath == 0 or (idx_ath + 1) == len(ath_galleries):
                                    print(f"      [{idx_ath+1}/{len(ath_galleries)}] Skipping {ath_name} (Already in DB)")
                                continue

                            print(f"      [{idx_ath+1}/{len(ath_galleries)}] Processing Athlete: {ath_name}...")
                            ath_url = ath["url"]
                            if not ath_url.startswith("http"):
                                ath_url = f"https://contests.npcnewsonline.com{ath_url}"
                            
                            # --- PAGINATION & LINK DISCOVERY ---
                            all_viewer_links = set()
                            current_ath_url = ath_url
                            
                            try:
                                while current_ath_url:
                                    page.goto(current_ath_url, wait_until="domcontentloaded", timeout=60000)
                                    page_links = page.locator("a[href*='images.php']").evaluate_all("""
                                        elements => elements.map(el => el.getAttribute('href'))
                                    """)
                                    for l in page_links:
                                        if l: all_viewer_links.add(l if l.startswith("http") else f"https://contests.npcnewsonline.com/{l.lstrip('/')}")
                                    
                                    next_page = page.locator("a:has-text('Next')").first
                                    if next_page.is_visible():
                                        next_href = next_page.get_attribute("href")
                                        current_ath_url = next_href if next_href.startswith("http") else f"https://contests.npcnewsonline.com/{next_href.lstrip('/')}"
                                    else:
                                        current_ath_url = None
                            except Exception as e:
                                print(f"      [Warning] Navigation issue for {ath_name}: {e}. Proceeding with found links...")

                            unique_viewers = list(all_viewer_links)
                            image_targets = []
                            if unique_viewers:
                                with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"] * 2) as discovery_executor:
                                    results = list(discovery_executor.map(self._find_high_res_src, unique_viewers))
                                    d_clean = self._sanitize(division)
                                    a_clean = self._sanitize(ath_name)
                                    valid_results = [r for r in results if r]
                                    for idx, high_res_src in enumerate(valid_results):
                                        filename = f"{year_str}_{c_clean}_{d_clean}_{a_clean}_{idx+1}.jpg"
                                        image_targets.append((high_res_src, target_dir / filename))

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
                                # Queue up whatever is done, though since we upload per contest, 
                                # if a single contest breaches it we could upload mid-contest but let's wait until contest finishes
                                pass

                        if contest_new_images > 0:
                            self._async_upload_and_cleanup_contest(year, c_name, target_dir, backup_db=False)

                browser.close()
        finally:
            print("\n[Shutdown] Waiting for background uploads to complete...")
            self.upload_executor.shutdown(wait=True)
            self.db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Specific year to scrape")
    args = parser.parse_args()
    scraper = NPCNewsScraper([args.year] if args.year else list(range(2026, 2014, -1)))
    scraper.run()