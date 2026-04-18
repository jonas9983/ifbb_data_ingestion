import os
import subprocess
import argparse
import sys

def upload_to_drive(source_path: str, remote_destination: str):
    """
    Syncs a local directory to a remote cloud destination using Rclone.
    
    Args:
        source_path (str): The local directory to upload (e.g., 'data/images')
        remote_destination (str): The rclone remote and path (e.g., 'gdrive:Bodybuilding_Dataset')
    """
    if not os.path.exists(source_path):
        print(f" Error: The source path '{source_path}' does not exist.")
        sys.exit(1)

    print(f" Starting Rclone Upload")
    print(f" Source: {source_path}")
    print(f" Destination: {remote_destination}")

    try:
        # Run the rclone copy command with real-time output
        subprocess.run(
            ["rclone", "copy", source_path, remote_destination, "--progress"],
            check=True
        )
        print("\n Upload successfully completed!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n Upload failed. Rclone encountered an error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n Rclone is not installed or not in your system's PATH.")
        print("Please install it using: curl https://rclone.org/install.sh | sudo bash")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n Upload cancelled by user.")
        sys.exit(0)

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
    
    args = parser.parse_args()
    
    # Execute the upload
    upload_to_drive(args.source, args.remote)