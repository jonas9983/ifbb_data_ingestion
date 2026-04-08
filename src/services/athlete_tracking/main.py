import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
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
        
        _ = tracker.process_frame(img, frame_name, frame_number, show_person_bbox=True, filter_front_row=True)
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f" -> Processed {processed_count} frames...")
        
        if args.save_video:
            if video_writer is None:
                h, w = img.shape[:2]
                out_path = os.path.join(args.output_dir, "tracking_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                print(f"🎥 Saving video to {out_path}")
            video_writer.write(img)

        if args.debug:
            view_img = cv2.resize(img, (1280, 720)) if img.shape[1] > 1280 else img
            cv2.imshow("Athlete Tracking", view_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early exit requested by user.")
                break

    # 4. Cleanup and Export
    if video_writer:
        video_writer.release()

    print("\n" + "="*60)
    print(f" PROCESSING COMPLETE. Processed {processed_count} frames.")
    print("="*60)
    
    tracker.print_summary()
    tracker.export_comprehensive_data(os.path.join(args.output_dir, "comprehensive_tracking_data.json"))
    
    if args.debug:
        cv2.destroyAllWindows()

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
    parser.add_argument("--debug", action="store_true", help="Enable live preview")

    args = parser.parse_args()
    main(args)