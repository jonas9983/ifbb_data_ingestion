import os
import subprocess
import argparse
import zipfile
import shutil
from pathlib import Path

def download_from_drive(remote_path: str, local_destination: str):
    """
    Downloads a file or directory from a remote cloud destination using Rclone.
    
    Args:
        remote_path (str): The rclone remote and path (e.g., 'gdrive:Bodybuilding_Dataset/npc_data.db')
        local_destination (str): The local path to save the file (e.g., 'data/npc_data.db')
    """
    print(f" Starting Rclone Download")
    print(f" Remote Source: {remote_path}")
    print(f" Local Destination: {local_destination}")

    # Ensure local directory exists if downloading a file
    local_path = Path(local_destination)
    if not local_path.suffix: # It's a directory
        local_path.mkdir(parents=True, exist_ok=True)
    else: # It's a file
        local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use copyto for single files, copy for directories
        mode = "copyto" if local_path.suffix else "copy"
        
        # Run rclone command
        rclone_cmd = [
            "rclone", mode, remote_path, local_destination,
            "--update",
            "--transfers", "16",
            "--checkers", "32",
            "--fast-list",
            "--stats", "10s",
            "--stats-one-line"
        ]
        subprocess.run(rclone_cmd, check=True)
        print(f"\n Download successfully completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n Download failed. Rclone encountered an error: {e}")
        return False
    except FileNotFoundError:
        print("\n Rclone is not installed or not in your system's PATH.")
        return False
    except KeyboardInterrupt:
        print(f"\n\n Download cancelled by user.")
        return False

def fetch_and_extract_zip(remote_zip_path: str, local_extract_dir: str):
    """
    Downloads a ZIP from Drive and extracts it.
    
    Args:
        remote_zip_path (str): Rclone path to ZIP
        local_extract_dir (str): Directory to extract into
    """
    local_zip = Path("data/tmp_download.zip")
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    
    if download_from_drive(remote_zip_path, str(local_zip)):
        print(f" Extracting {local_zip} to {local_extract_dir}...")
        try:
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(local_extract_dir)
            local_zip.unlink()
            print(" Extraction complete.")
            return True
        except Exception as e:
            print(f" Extraction failed: {e}")
            return False
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Google Drive using Rclone.")
    parser.add_argument("--remote", type=str, required=True, help="Remote source path")
    parser.add_argument("--local", type=str, required=True, help="Local destination path")
    parser.add_argument("--extract", action="store_true", help="Extract if it's a ZIP")
    args = parser.parse_args()

    if args.extract:
        fetch_and_extract_zip(args.remote, args.local)
    else:
        download_from_drive(args.remote, args.local)
