import os
import subprocess
import argparse
import sys

def upload_to_drive(source_path: str, remote_destination: str, delete_after: bool = False):
    """
    Syncs a local directory to a remote cloud destination using Rclone.
    
    Args:
        source_path (str): The local directory to upload (e.g., 'data/images')
        remote_destination (str): The rclone remote and path (e.g., 'gdrive:Bodybuilding_Dataset')
        delete_after (bool): If True, uses 'rclone move' to delete local files after successful upload.
    """
    if not os.path.exists(source_path):
        print(f" Error: The source path '{source_path}' does not exist.")
        return

    command = "move" if delete_after else "copy"
    print(f" Starting Rclone {command.capitalize()}")
    print(f" Source: {source_path}")
    print(f" Destination: {remote_destination}")

    try:
        # Run the rclone command with optimized flags for speed and reduced verbosity
        # --progress allows the user to see the status during long moves
        rclone_cmd = [
            "rclone", command, source_path, remote_destination, 
            "--transfers", "4", 
            "--checkers", "8", 
            "--drive-chunk-size", "64M",
            "--fast-list",
            "--progress",
            "--stats", "15s"
        ]
        subprocess.run(rclone_cmd, check=True)
        print(f"\n {command.capitalize()} successfully completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n {command.capitalize()} failed. Rclone encountered an error: {e}")
    except FileNotFoundError:
        print("\n Rclone is not installed or not in your system's PATH.")
        print("Please install it using: curl https://rclone.org/install.sh | sudo bash")
    except KeyboardInterrupt:
        print(f"\n\n {command.capitalize()} cancelled by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local directories to Google Drive using Rclone.")
    
    # Define the command-line arguments
    parser.add_argument(
        "--source", 
        type=str, 
        default="data/images",
        help="The local directory you want to upload (default: data/images)"
    )
    
    parser.add_argument(
        "--remote", 
        type=str, 
        default="gdrive:Bodybuilding_Dataset",
        help="The rclone remote and destination folder (default: gdrive:Bodybuilding_Dataset)"
    )

    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="Use rclone move to delete local files after successful upload"
    )

    args = parser.parse_args()

    # Execute the upload
    upload_to_drive(args.source, args.remote, args.delete_after)