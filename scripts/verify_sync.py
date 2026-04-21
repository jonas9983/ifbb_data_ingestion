import sqlite3
import subprocess
from pathlib import Path

# --- CONFIG ---
DB_PATH = "data/npc_database.db"
GDRIVE_REMOTE = "gdrive:personal/Bodybuilding_Dataset"

def get_remote_zips(year):
    """Lists zips in the remote year folder."""
    remote_path = f"{GDRIVE_REMOTE}/{year}"
    try:
        result = subprocess.run(
            ["rclone", "lsf", remote_path, "--files-only"],
            capture_output=True, text=True, check=True
        )
        return [f.strip() for f in result.stdout.splitlines() if f.endswith(".zip")]
    except subprocess.CalledProcessError as e:
        print(f"  [Warning] Could not list remote directory for {year}: {e}")
        return []
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [Error] Unexpected error: {e}")
        return []

def verify_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Checking database for contests...")
    cursor.execute("SELECT DISTINCT year, contest_name FROM athletes ORDER BY year DESC")
    contests = cursor.fetchall()
    
    missing_contests = []
    
    # Cache remote files per year
    remote_cache = {}
    
    for year, contest_name in contests:
        if year not in remote_cache:
            print(f"Fetching remote list for {year}...")
            remote_cache[year] = get_remote_zips(year)
        
        # Format the expected zip name (same as main.py _sanitize)
        c_clean = "".join(c for c in contest_name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        zip_name = f"{year}_{c_clean}.zip"
        
        if zip_name not in remote_cache[year]:
            print(f"!!! MISSING ON DRIVE: {year} - {contest_name}")
            missing_contests.append((year, contest_name))

    if missing_contests:
        print(f"\nFound {len(missing_contests)} contests in DB that are NOT on Drive.")
        print("To fix this, you should delete these from the DB so the scraper can try again:")
        for year, name in missing_contests:
            print(f"DELETE FROM athletes WHERE year = {year} AND contest_name = '{name}';")
    else:
        print("\nAll good! Everything in the DB has a matching zip on Drive.")
    
    conn.close()

if __name__ == "__main__":
    verify_data()
