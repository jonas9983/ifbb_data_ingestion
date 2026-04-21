import os
import argparse
import subprocess
from pathlib import Path
from ifbb_data_ingestion.etl.loading.drive_loading import upload_to_drive

def download_youtube_video(url: str, output_dir: Path):
    """
    Downloads a YouTube video using yt-dlp.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Template for the output filename
    # %(title)s - %(id)s.%(ext)s
    output_template = str(output_dir / "%(title)s.%(ext)s")
    
    print(f"[*] Downloading video from: {url}")
    
    command = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", # Prefer mp4
        "--merge-output-format", "mp4",
        "-o", output_template,
        url
    ]
    
    try:
        # We use check=True to raise an error if the download fails
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("[+] Download completed successfully.")
        
        # Find the downloaded file (yt-dlp doesn't return the filename easily in simple mode)
        # We'll look for the most recently created file in the output directory
        files = list(output_dir.glob("*.mp4"))
        if not files:
            # Try other extensions just in case
            files = list(output_dir.glob("*"))
            
        if files:
            # Sort by modification time to get the latest
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
        return None

    except subprocess.CalledProcessError as e:
        print(f"[-] Error downloading video: {e.stderr}")
        return None
    except FileNotFoundError:
        print("[-] Error: 'yt-dlp' not found. Please install it with: pip install yt-dlp")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download a YouTube video and upload it to Google Drive.")
    parser.add_argument("url", help="The YouTube video URL")
    parser.add_argument("--remote", default="gdrive:personal/Bodybuilding_Dataset/Videos",
                        help="The rclone remote destination (default: gdrive:personal/Bodybuilding_Dataset/Videos)")
    parser.add_argument("--keep", action="store_true", help="Keep the local file after upload")
    
    args = parser.parse_args()
    
    # Temporary local storage
    temp_dir = Path("data/temp_videos")
    
    downloaded_file = download_youtube_video(args.url, temp_dir)
    
    if downloaded_file and downloaded_file.exists():
        print(f"[*] Uploading {downloaded_file.name} to {args.remote}...")
        
        # Use move if we don't want to keep the file, otherwise copy
        # Our upload_to_drive helper handles this via delete_after
        upload_to_drive(str(downloaded_file), args.remote, delete_after=not args.keep)
        
        # Cleanup temp dir if empty
        try:
            if not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except:
            pass
    else:
        print("[-] Failed to download video. Upload aborted.")

if __name__ == "__main__":
    main()
