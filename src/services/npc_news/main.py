import os
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
CONFIG = {
    "BASE_URL": "https://contests.npcnewsonline.com/contests/",
    "STORAGE_BASE": "data/images",
    "METADATA_FILE": "data/images/metadata.json",
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
            res = self.img_session.get(url, timeout=10)
            if res.status_code == 200:
                # LAZY CREATION: Only make the folder if we actually have an image to save
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
            page.goto("https://contests.npcnewsonline.com/", wait_until="networkidle")
            page.wait_for_timeout(2000)

            for year in self.years:
                year_str = str(year)
                print(f"\n--- Processing Year: {year_str} ---")
                self.metadata[year_str] = {}
                
                year_url = f"{CONFIG['BASE_URL']}{year}/"
                page.goto(year_url, wait_until="networkidle")
                
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

                    page.goto(year_url, wait_until="networkidle")
                    link_to_click = page.locator(f"a[href='{contest['href']}']").first
                    
                    try:
                        link_to_click.click()
                        page.wait_for_load_state("networkidle")
                        page.wait_for_timeout(2000)
                    except Exception as e:
                        print(f"    Click failed for {c_name}. Retrying with goto...")
                        page.goto(contest["href"], wait_until="networkidle")

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

                    print(f"    Found {len(ath_galleries)} links to process...")

                    for ath in ath_galleries:
                        ath_name = ath["name"]
                        ath_url = ath["url"]
                        if not ath_url.startswith("http"):
                            ath_url = f"https://contests.npcnewsonline.com{ath_url}"
                        
                        target_dir = Path(CONFIG["STORAGE_BASE"]) / year_str / self._sanitize(c_name) / "General" / self._sanitize(ath_name)
                        
                        # Navigate to athlete gallery
                        page.goto(ath_url, wait_until="networkidle")
                        page.wait_for_timeout(1000)
                        
                        img_sources = page.locator("img").evaluate_all("""
                            elements => elements.map(el => el.getAttribute('src') || el.getAttribute('data-src') || '')
                        """)
                        
                        successful_images = 0
                        for idx, src in enumerate(img_sources):
                            if src and ("/images/contests/" in src or "thumb" in src):
                                if not src.startswith("http"): src = f"https://contests.npcnewsonline.com{src}"
                                
                                # 1. Attempt to guess the high-res URL by removing 'thumb_' or '-th'
                                high_res_url = src.replace("thumb_", "").replace("_thumb", "").replace("-th.", ".")
                                filename = f"image_{idx+1}.jpg"
                                
                                # 2. Try high-res first. If it fails, fallback to the thumbnail.
                                if self.download_image(high_res_url, target_dir / filename) or self.download_image(src, target_dir / filename):
                                    self.metadata[year_str][c_name].setdefault("General", {}).setdefault(ath_name, []).append(filename)
                                    successful_images += 1
                        
                        if successful_images > 0:
                            print(f"      Athlete: {ath_name} -> Done: {successful_images} images")

                self._save_metadata()
            browser.close()

    def _save_metadata(self):
        with open(CONFIG["METADATA_FILE"], "w") as f:
            json.dump(self.metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    scraper = NPCNewsScraper([args.year] if args.year else list(range(2011, 2027)))
    scraper.run()