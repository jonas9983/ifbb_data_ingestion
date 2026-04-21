import os
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoFrameExtractor:
    """
    Iterates over a specified folder (or all user folders), finds video files, 
    and extracts 1 frame per second (1fps) into a dedicated output folder for each video.
    """
    
    def __init__(self, base_dir="instagram_downloads", target_user=None, workers=4):
        """
        Initializes the VideoFrameExtractor.

        Args:
            base_dir (str): The root directory where user folders are located.
            target_user (str | None): A specific username folder to process.
            workers (int): Number of parallel video processing threads.
        """
        self.base_dir = Path(base_dir)
        # If a target user is specified, update the directory to search within
        if target_user:
            self.search_dir = self.base_dir / target_user
            print(f"Targeting specific user folder: {self.search_dir}")
        else:
            self.search_dir = self.base_dir
            
        self.workers = workers
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'extracted_frames': 0
        }

    def _extract_frames_ffmpeg(self, video_path):
        """
        Calls FFmpeg to extract frames at 1fps and organizes the output.
        
        Args:
            video_path (Path): Full path to the input video file.
            
        Returns:
            tuple: (success (bool), extracted_frames_count (int))
        """
        # Create a unique output folder for the video frames
        video_filename_stem = video_path.stem 
        output_dir = video_path.parent / f"{video_filename_stem}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command setup (remains the same)
        command = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', 'fps=1',
            '-q:v', '2',
            str(output_dir / 'frame_%04d.jpg')
        ]

        print(f"  -> Processing: {video_path.name} to {output_dir.name}")

        try:
            # Run the FFmpeg command
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Count the extracted frames
            frame_count = len(list(output_dir.glob('frame_*.jpg')))
            
            return True, frame_count
            
        except subprocess.CalledProcessError as e:
            print(f"  FFmpeg Error for {video_path.name}: {e.stderr.strip()}")
            return False, 0
        except FileNotFoundError:
             print("  Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH.")
             return False, 0
        except Exception as e:
            print(f"  Unknown Error for {video_path.name}: {e}")
            return False, 0

    def run(self):
        """Main method to find and process all videos."""
        
        if not self.search_dir.is_dir():
            print(f"Error: Target directory not found at {self.search_dir}")
            return
            
        print(f"Searching for videos in: {self.search_dir}")
        
        # 1. Collect all video files. If target_user is set, this searches only that folder.
        video_files = list(self.search_dir.rglob('*_video.mp4'))
        
        if not video_files:
            print("No video files (*_video.mp4) found in the target directory.")
            return

        self.stats['total_videos'] = len(video_files)
        print(f"Found {self.stats['total_videos']} videos to process.")
        print(f"Starting parallel processing with {self.workers} workers...")
        print("-" * 50)

        # 2. Process videos in parallel (remaining logic is unchanged)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_video = {
                executor.submit(self._extract_frames_ffmpeg, video_path): video_path
                for video_path in video_files
            }

            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                
                try:
                    success, frame_count = future.result()
                    
                    if success:
                        self.stats['processed_videos'] += 1
                        self.stats['extracted_frames'] += frame_count
                        print(f"  SUCCESS: {video_path.name} -> Extracted {frame_count} frames.")
                    else:
                        self.stats['failed_videos'] += 1
                        print(f"  FAILED: {video_path.name}")
                        
                except Exception as exc:
                    print(f"  Critical error processing {video_path.name}: {exc}")
                    self.stats['failed_videos'] += 1
        
        # 3. Print summary
        self._print_summary()

    def _print_summary(self):
        """Prints the final processing summary."""
        print("\n" + "="*60)
        print("VIDEO FRAME EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total videos found: {self.stats['total_videos']}")
        print(f"Videos successfully processed: {self.stats['processed_videos']}")
        print(f"Total frames extracted (1fps): {self.stats['extracted_frames']}")
        print(f"Videos failed to process: {self.stats['failed_videos']}")
        print(f"\nFrames saved in new subdirectories (e.g., user_folder/video_ID_frames)")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract 1 frame per second from all videos in user download folders.'
    )
    parser.add_argument(
        '-d', '--directory',
        default='instagram_downloads',
        help='Base directory containing the user folders (default: instagram_downloads)'
    )
    parser.add_argument(
        '-u', '--user',
        default=None,
        help='Optional: Specify a single username subfolder to process (e.g., fabriciomoreirapro)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel video processing workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    processor = VideoFrameExtractor(
        base_dir=args.directory,
        target_user=args.user,
        workers=args.workers
    )
    
    processor.run()