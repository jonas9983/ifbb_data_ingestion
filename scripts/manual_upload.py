import argparse
from pathlib import Path
from src.etl.loading.drive_loading import upload_to_drive

def trigger_upload(staging_dir_name="1776657595_all", remote="gdrive:personal/Bodybuilding_Dataset"):
    staging_path = Path("data/upload_staging") / staging_dir_name
    
    if not staging_path.exists():
        print(f"Error: Staging directory not found at {staging_path}")
        return

    print(f"Triggering manual upload of {staging_path} to {remote}...")
    upload_to_drive(str(staging_path), remote, delete_after=True)
    
    # Cleanup empty folder
    if staging_path.exists() and not any(staging_path.iterdir()):
        try:
            staging_path.rmdir()
            print("Successfully cleaned up empty staging directory.")
        except Exception as e:
            print(f"Could not remove directory: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually trigger an upload from a specific staging folder.")
    parser.add_argument("--folder", type=str, default="1776657595_all", help="The name of the folder inside data/upload_staging")
    args = parser.parse_args()
    
    trigger_upload(staging_dir_name=args.folder)
