import os
import subprocess
import argparse
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
        # Run rclone copy from remote to local
        # --update: skip files that are newer on the destination
        rclone_cmd = [
            "rclone", "copy", remote_path, local_destination,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Google Drive using Rclone.")
    parser.add_argument("--remote", type=str, required=True, help="Remote source path")
    parser.add_argument("--local", type=str, required=True, help="Local destination path")
    args = parser.parse_args()

    download_from_drive(args.remote, args.local)
