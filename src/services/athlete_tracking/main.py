import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import time
import argparse
import threading
from queue import Queue
from src.services.athlete_tracking.tracker_pipeline import AthletePositionTracker
from src.services.helpers.video_utils import parse_frame_range, get_video_frames

# --- THREAD WORKER: READS VIDEO ---
def video_reader_worker(video_path, start_f, end_f, step_f, input_queue):
    for frame_number, frame_name, img in get_video_frames(video_path, start_f, end_f, step_f):
        # max_height resizing done here so the GPU thread doesn't have to waste time doing it
        max_height = 720
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))
            
        input_queue.put((frame_number, frame_name, img))
    
    input_queue.put(None)

# --- THREAD WORKER: WRITES VIDEO ---
def video_writer_worker(output_path, output_queue):
    writer = None
    while True:
        data = output_queue.get()
        if data is None:
            break
            
        img = data
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
            print(f" Saving video to {output_path} at {w}x{h} resolution")
            
        writer.write(img)
        
    if writer:
        writer.release()


def main(args):
    if not os.path.exists(args.video_path):
        print(f"\n ERROR: Video file not found at {args.video_path}")
        return

    start_f, end_f, step_f = parse_frame_range(args.frame_range)

    print("\n--- INITIALIZING MODELS ---")
    tracker = AthletePositionTracker(
        db_path=args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tracking_output.mp4")
    
    # --- QUEUE SETUP ---
    input_queue = Queue(maxsize=120) 
    output_queue = Queue(maxsize=120)

    # --- START THREADS ---
    reader_thread = threading.Thread(target=video_reader_worker, args=(args.video_path, start_f, end_f, step_f, input_queue))
    reader_thread.start()

    writer_thread = None
    if args.save_video:
        writer_thread = threading.Thread(target=video_writer_worker, args=(out_path, output_queue))
        writer_thread.start()

    print(f"\n--- STARTING PROCESSING (THREADED) ---")
    processed_count = 0
    pipeline_start_time = time.time()
    
    # --- MAIN GPU LOOP ---
    while True:
        data = input_queue.get()
        if data is None: # Video is completely read
            break
            
        frame_number, frame_name, img = data
        start_time = time.time()
        
        # 1. Track & Detect (GPU Heavy)
        athletes = tracker.process_frame(img, frame_name, frame_number, show_person_bbox=True, filter_front_row=False)
        track_time = time.time() - start_time
        
        # 2. Print clean console output
        if args.debug or processed_count % 30 == 0:
            found_names = [a.name for a in athletes if a.name != "Unknown"]
            if len(athletes) == 0:
                print(f"Frame {frame_number} | No stage detected. | Track: {track_time:.2f}s | Queue: {input_queue.qsize()}")
            else:
                print(f"Frame {frame_number} | Tracked: {len(athletes)} | Recog: {found_names} | Track: {track_time:.2f}s")
        
        processed_count += 1
        
        # 3. Send to background writer
        if args.save_video:
            output_queue.put(img)

    # --- CLEANUP ---
    if args.save_video:
        output_queue.put(None)
        writer_thread.join()
        
    reader_thread.join()
        
    total_time = time.time() - pipeline_start_time
    fps = processed_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print(f" PROCESSING COMPLETE.")
    print(f" Processed {processed_count} frames in {total_time:.2f} seconds.")
    print(f" Average Speed: {fps:.2f} FPS")
    print("="*60)
    
    if processed_count > 0:
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