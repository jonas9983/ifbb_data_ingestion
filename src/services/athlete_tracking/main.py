import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import time
import argparse
from src.services.athlete_tracking.tracker_pipeline import AthletePositionTracker
from src.services.helpers.video_utils import parse_frame_range, download_frames_parallel, get_local_images

def main(args):
    start_f, end_f, step_f = parse_frame_range(args.frame_range)

    # 1. Ensure Data exists
    download_frames_parallel(args.base_url, args.local_cache_dir, start_f, end_f, step_f)

    # 2. Initialize Pipeline
    print("\n--- INITIALIZING MODELS ---")
    tracker = AthletePositionTracker(
        db_path=args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )

    os.makedirs(args.output_dir, exist_ok=True)
    video_writer = None
    
    # 3. Main Processing Loop
    print(f"\n--- STARTING PROCESSING ---")
    processed_count = 0
    
    for frame_number, frame_name, img in get_local_images(args.local_cache_dir, start_f, end_f, step_f):
        start_time = time.time()

        # Scale down for processing speed
        max_height = 1080
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
        
        # 1. Track & Detect
        athletes = tracker.process_frame(img, frame_name, frame_number, show_person_bbox=True, filter_front_row=True)
        track_time = time.time() - start_time
        
        found_names = [a.name for a in athletes if a.name not in ["Unknown", "MARSHALL"]]
        print(f"Frame {frame_number} | Bodies Tracked: {len(athletes)} | Recognized: {found_names} | Track time: {track_time:.2f}s")
        
        processed_count += 1
        
        # 3. Save Video
        vid_start_time = time.time()
        if args.save_video:
            if video_writer is None:
                h, w = img.shape[:2]
                out_path = os.path.join(args.output_dir, "tracking_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                print(f" Saving video to {out_path} at {w}x{h} resolution")
            video_writer.write(img)
            
        vid_time = time.time() - vid_start_time
        if args.save_video and args.debug:
             print(f" -> Video write time: {vid_time:.2f}s")

    # 4. Cleanup and Export
    if video_writer:
        video_writer.release()

    print("\n" + "="*60)
    print(f" PROCESSING COMPLETE. Processed {processed_count} frames.")
    print("="*60)
    
    tracker.print_summary()
    tracker.export_comprehensive_data(os.path.join(args.output_dir, "comprehensive_tracking_data.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athlete Tracker")
    parser.add_argument("--base_url", required=True, help="URL with {:05d} placeholder")
    parser.add_argument("--local_cache_dir", required=True, help="Where to save the downloaded frames")
    parser.add_argument("--db", required=True, help="Path to face database")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--save_video", action="store_true", help="Compile processed frames into an mp4 video")
    parser.add_argument("--threshold", type=float, default=0.35, help="Face recognition threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="Person detection confidence")
    parser.add_argument("--frame-range", type=str, default="1:100:1", help="Format: 'start:end:step'")
    parser.add_argument("--tracker-config", type=str, default=None, help="YOLO tracker config")
    parser.add_argument("--debug", action="store_true", help="Enable console debug logs")

    args = parser.parse_args()
    main(args)