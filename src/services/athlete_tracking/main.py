import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import json
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import queue
import requests
from datetime import datetime
from src.services.faces.face_recognizer import FaceRecognizer
from src.services.athlete_tracking.detection import AthleteDetector
from src.services.athlete_tracking.swap_detection import SwapDetector
from src.services.athlete_tracking.yolo_segmentation import YOLOSegmentationModel
from src.services.helpers.frame_logger import FrameLogger

class AthletePositionTracker:
    def __init__(
        self,
        db_path: str,
        threshold: float = 0.35,
        confidence_threshold: float = 0.5,
        tracker_config: str = None,
        debug_mode: bool = False
    ):
        print("Loading FaceRecognizer...")
        face_recognizer = FaceRecognizer(db_path, threshold=threshold)
        print("Recognizer loaded.")
        
        person_model = YOLOSegmentationModel(
            model_path='yolo11l-seg.pt',
            tracker_config=tracker_config
        )
        print("Person segmentation model loaded.")
        
        self.detector = AthleteDetector(
            face_recognizer,
            person_model,
            confidence_threshold=confidence_threshold,
            debug_mode=debug_mode 
        )
        self.swap_detector = SwapDetector()
        self.frame_logger = FrameLogger()
        
        self.last_gray_frame: Optional[np.ndarray] = None
        self.cut_threshold: float = 25.0

    def _check_for_camera_cut(self, img: np.ndarray, frame_name: str) -> Tuple[bool, float]:
        """
        Detects camera cuts and returns (is_cut, avg_diff).
        """
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.last_gray_frame is None:
            self.last_gray_frame = current_gray
            return False, 0.0

        diff = cv2.absdiff(current_gray, self.last_gray_frame)
        avg_diff = float(np.mean(diff))
        
        is_cut = avg_diff > self.cut_threshold
        
        self.last_gray_frame = current_gray
        
        if is_cut:
            print(f" CAMERA CUT: {frame_name} (diff={avg_diff:.2f})")
            
            # Reset tracking
            if hasattr(self.detector.person_detector.model, 'predictor') and \
               hasattr(self.detector.person_detector.model.predictor, 'trackers'):
                for tracker in self.detector.person_detector.model.predictor.trackers:
                    tracker.reset()
            
            self.detector.athlete_registry.clear()
            self.detector.marshall_track_id = -1
            
        return is_cut, avg_diff

    def process_frame(
        self, 
        img, 
        frame_name: str,
        frame_number: int,
        show_person_bbox: bool = True,
        filter_front_row: bool = True
    ):
        """
        Process frame and log comprehensive data.
        """
        # Camera cut detection
        is_cut, avg_diff = self._check_for_camera_cut(img, frame_name)
        
        if is_cut:
            self.swap_detector.reset_state(frame_number, frame_name, "camera_cut")
        
        # Store raw face detections BEFORE association
        raw_faces = self.detector.detect_faces(img)
        
        # Detect and associate athletes
        all_athletes = self.detector.detect_and_associate(img, check_depth=filter_front_row)
        
        # Filter for swap detection
        valid_front_row_athletes = [
            a for a in all_athletes 
            if a.is_front_row and a.name != "Unknown" and a.name != "MARSHALL"
        ]
        
        current_positions = {
            athlete.name: athlete.center_x 
            for athlete in valid_front_row_athletes
        }
        
        # Update swap detector (THIS NOW LOGS EVERY FRAME)
        swap_info = self.swap_detector.update_state(
            current_positions,
            frame_name,
            frame_number
        )
        
        # Add positions to swap_info for logging
        swap_info['positions'] = current_positions
        
        # === LOG COMPREHENSIVE FRAME DATA ===
        self.frame_logger.log_frame(
            frame_number=frame_number,
            frame_name=frame_name,
            athletes=all_athletes,
            raw_faces=raw_faces,
            is_camera_cut=is_cut,
            avg_pixel_diff=avg_diff,
            registry_state=self.detector.athlete_registry.copy(),
            marshall_track_id=self.detector.marshall_track_id,
            swap_info=swap_info,
            frame_dims=img.shape[:2]
        )
        
        # Visualize
        self.detector.draw_annotations(img, all_athletes, show_person_bbox)
        
        return all_athletes
    
    def export_events(self, output_path: str):
        """Export swap events (legacy format)."""
        stats = self.swap_detector.get_statistics()
        
        events_data = {
            'summary': stats,
            'events': [event.to_dict() for event in self.swap_detector.events]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Events exported to: {output_path}")
    
    def export_comprehensive_data(self, output_path: str):
        """
        Export comprehensive frame-by-frame data.
        This is the MAIN export you'll use for post-processing.
        """
        self.frame_logger.export_to_json(output_path)
    
    def print_summary(self):
        """Print tracking summary."""
        self.swap_detector.print_summary()


def parse_frame_range(range_str: str) -> Tuple[Optional[int], Optional[int], int]:
    """Parse frame range string in format 'start:end:step'."""
    if not range_str: return 1, 100, 1 # Default to first 100 frames if none provided
    parts = range_str.split(':')
    start = int(parts[0]) if len(parts) > 0 and parts[0] else 1
    end = int(parts[1]) if len(parts) > 1 and parts[1] else 100
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return start, end, step

def download_frames_parallel(base_url: str, save_dir: str, start_f: int, end_f: int, step: int = 1):
    """Downloads frames heavily in parallel before tracking begins. Skips existing files."""
    os.makedirs(save_dir, exist_ok=True)
    indices = list(range(start_f, end_f + 1, step))
    
    print(f"\n--- Checking/Downloading {len(indices)} frames to {save_dir} ---")
    
    def fetch(idx):
        filename = f"shots_{idx:05d}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Skip if we already downloaded it
        if os.path.exists(filepath):
            return True
            
        url = base_url.format(idx)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
        return False

    # Download using 8 concurrent threads for max speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch, indices))
        
    success_count = sum(results)
    if success_count < len(indices):
        print(f" Warning: Only downloaded {success_count}/{len(indices)} frames successfully.")
    else:
        print(f" All {success_count} frames ready locally!")

def get_local_images(input_dir: str, start_f: int, end_f: int, step: int = 1):
    """Reads images directly from the local drive."""
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")) + 
                         glob.glob(os.path.join(input_dir, "*.jpg")))
    
    for path in image_paths:
        try:
            frame_number = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        except ValueError:
            continue

        if frame_number < start_f or frame_number > end_f or frame_number % step != 0: 
            continue

        img = cv2.imread(path)
        if img is not None:
            yield frame_number, os.path.basename(path), img

def main(args):
    start_f, end_f, step_f = parse_frame_range(args.frame_range)

    # DOWNLOAD THE BATCH FIRST
    download_frames_parallel(args.base_url, args.local_cache_dir, start_f, end_f, step_f)

    # INITIALIZE TRACKER
    print("\n--- INITIALIZING AI MODELS ---")
    tracker = AthletePositionTracker(
        args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )

    os.makedirs(args.output_dir, exist_ok=True)
    video_writer = None
    
    print(f"\n--- STARTING PROCESSING ---")
    processed_count = 0
    
    for frame_number, frame_name, img in get_local_images(args.local_cache_dir, start_f, end_f, step_f):
        
        all_athletes = tracker.process_frame(
            img, frame_name, frame_number,
            show_person_bbox=True, filter_front_row=True
        )
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f" Processed {processed_count} frames...")
        
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

    if video_writer:
        video_writer.release()

    print("\n" + "="*60)
    print(f" PROCESSING COMPLETE. Processed {processed_count} frames.")
    print("="*60)
    
    tracker.print_summary()
    comprehensive_path = os.path.join(args.output_dir, "comprehensive_tracking_data.json")
    tracker.export_comprehensive_data(comprehensive_path)
    
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