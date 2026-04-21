import os
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

# --- CONFIG ---
SOURCE_REMOTE = "gdrive:personal/Bodybuilding_Dataset_test"
DEST_REMOTE = "gdrive:personal/Bodybuilding_Dataset"
TEMP_DIR = Path("data/migration_temp")

def get_remote_files(remote_path):
    """Lists files in a remote directory using rclone."""
    try:
        result = subprocess.run(
            ["rclone", "lsf", remote_path, "--files-only"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.splitlines()
    except Exception as e:
        print(f"Error listing {remote_path}: {e}")
        return []

def migrate_year(year):
    print(f"\n>>> Migrating Year: {year}")
    remote_year_path = f"{SOURCE_REMOTE}/{year}"
    files = get_remote_files(remote_year_path)
    
    if not files:
        print(f"No files found for year {year}")
        return

    # Group files by contest
    # Pattern: [Year]_[Contest]_[Division]_[Athlete]_[Index].jpg
    # Since contest names can have underscores, we'll try to find common prefixes
    contests = defaultdict(list)
    for f in files:
        if not f.endswith(".jpg"): continue
        # Basic heuristic: the first few parts before division/athlete/index
        # Most files seem to follow: 2013_Arnold_Sports_Festival_...
        # We'll group by the part after the year but before the specific athlete details if possible
        # For now, let's take everything between the first underscore and the last 3 underscores
        parts = f.split("_")
        if len(parts) > 4:
            # Join parts to reconstruct contest name - this is a bit fuzzy but works if naming is consistent
            # Usually: [Year]_[Contest]_[Rest]
            # Let's just group by the year and first 3 words of the contest for safety, or prompt-based
            contest_guess = "_".join(parts[1:4]) 
            contests[contest_guess].append(f)
        else:
            contests["Other"].append(f)

    print(f"Found {len(contests)} potential contests in {year}")

    for contest_name, contest_files in contests.items():
        if contest_name == "Other": continue
        
        print(f"  Processing {contest_name} ({len(contest_files)} files)...")
        contest_temp = TEMP_DIR / str(year) / contest_name
        contest_temp.mkdir(parents=True, exist_ok=True)

        # 1. Download files for this contest
        # We use a filter file to avoid multiple rclone calls
        filter_file = TEMP_DIR / "rclone_filter.txt"
        with open(filter_file, "w") as f:
            for cf in contest_files:
                f.write(f"+ {cf}\n")
            f.write("- *\n")

        try:
            subprocess.run([
                "rclone", "copy", remote_year_path, str(contest_temp),
                "--filter-from", str(filter_file),
                "--transfers", "16", "--progress"
            ], check=True)

            # 2. Zip the directory
            zip_name = f"{year}_{contest_name}"
            zip_path = TEMP_DIR / f"{zip_name}.zip"
            shutil.make_archive(str(TEMP_DIR / zip_name), 'zip', str(contest_temp))

            # 3. Upload Zip to Destination
            dest_path = f"{DEST_REMOTE}/{year}"
            print(f"    Uploading {zip_name}.zip to {dest_path}...")
            subprocess.run([
                "rclone", "move", str(zip_path), dest_path,
                "--drive-chunk-size", "64M"
            ], check=True)

            # 4. Cleanup local temp
            shutil.rmtree(contest_temp)
            print(f"    Success: {zip_name} migrated.")

        except Exception as e:
            print(f"    Error processing {contest_name}: {e}")

if __name__ == "__main__":
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    # Start with one year to test
    for y in [2013, 2014, 2012, 2011]:
        migrate_year(y)
    
    # Cleanup filter file
    if (TEMP_DIR / "rclone_filter.txt").exists():
        (TEMP_DIR / "rclone_filter.txt").unlink()
