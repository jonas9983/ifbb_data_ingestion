import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import time
import argparse
from src.services.athlete_tracking.tracker_pipeline import AthletePositionTracker
from src.services.helpers.video_utils import parse_frame_range, get_video_frames

def main(args):
    start_f, end_f, step_f = parse_frame_range(args.frame_range)

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
    pipeline_start_time = time.time()
    
    for frame_number, frame_name, img in get_video_frames(args.video_path, start_f, end_f, step_f):
        start_time = time.time()

        # Scale down for processing speed
        max_height = 720
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
        
        # 1. Track & Detect
        athletes = tracker.process_frame(img, frame_name, frame_number, show_person_bbox=True, filter_front_row=False)
        track_time = time.time() - start_time
        
        # 2. Print clean console output
        found_names = [a.name for a in athletes if a.name not in ["Unknown", "MARSHALL"]]
        
        if len(athletes) == 0:
            print(f"Frame {frame_number} | No stage detected. Skipped. | Track time: {track_time:.2f}s")
        else:
            print(f"Frame {frame_number} | Bodies Tracked: {len(athletes)} | Recognized: {found_names} | Track time: {track_time:.2f}s")
        
        processed_count += 1
        
        # 3. Save Video
        if args.save_video:
            if video_writer is None:
                h, w = img.shape[:2]
                out_path = os.path.join(args.output_dir, "tracking_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                print(f" Saving video to {out_path} at {w}x{h} resolution")
            video_writer.write(img)

    # 4. Cleanup and Export
    if video_writer:
        video_writer.release()
        
    total_time = time.time() - pipeline_start_time
    fps = processed_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print(f" PROCESSING COMPLETE.")
    print(f" Processed {processed_count} frames in {total_time:.2f} seconds.")
    print(f" Average Speed: {fps:.2f} FPS")
    print("="*60)
    
    tracker.print_summary()
    tracker.export_comprehensive_data(os.path.join(args.output_dir, "comprehensive_tracking_data.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athlete Tracker")
    parser.add_argument("--video_path", required=True, help="Path to the input mp4 video file")
    parser.add_argument("--db", required=True, help="Path to face database")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--save_video", action="store_true", help="Compile processed frames into an mp4 video")
    parser.add_argument("--threshold", type=float, default=0.35, help="Face recognition threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="Person detection confidence")
    parser.add_argument("--frame-range", type=str, default=None, help="Format: 'start:end:step'. Leave blank for all.")
    
    parser.add_argument("--tracker-config", type=str, default=None, help="YOLO tracker config")
    parser.add_argument("--debug", action="store_true", help="Enable console debug logs")

    args = parser.parse_args()
    main(args)