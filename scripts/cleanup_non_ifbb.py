import sqlite3
import os
from pathlib import Path

def cleanup():
    db_path = "data/npc_data.db"
    storage_base = Path("data/npc_news")
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Identify records to delete
    # Based on the user's request, we want to exclude NPC, NPC Worldwide, and CPA
    # We'll search for these in the contest_name
    patterns = ["%NPC%", "%NPC Worldwide%", "%CPA%"]
    
    total_deleted = 0
    for pattern in patterns:
        # Get filenames to delete from disk
        cur.execute("SELECT year, image_filename FROM athletes WHERE contest_name LIKE ?", (pattern,))
        records = cur.fetchall()
        
        if not records:
            continue
            
        print(f"Found {len(records)} records matching pattern '{pattern}'.")
        
        for year, filename in records:
            file_path = storage_base / str(year) / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    # print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Delete from DB
        cur.execute("SELECT count(*) FROM athletes WHERE contest_name LIKE ?", (pattern,))
        count = cur.fetchone()[0]
        cur.execute("DELETE FROM athletes WHERE contest_name LIKE ?", (pattern,))
        total_deleted += count
        print(f"Deleted {count} records from DB for pattern '{pattern}'.")

    conn.commit()
    conn.close()
    print(f"\nCleanup finished. Total records removed: {total_deleted}")

if __name__ == "__main__":
    cleanup()
