"""
Logs ALL detection data, associations, and events to enable
robust downstream filtering and validation.
"""

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
            print(f"🎬 CAMERA CUT: {frame_name} (diff={avg_diff:.2f})")
            
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
        
        # Update swap detector
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
        
        print(f"\n📊 Events exported to: {output_path}")
    
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
    parts = range_str.split(':')
    
    if len(parts) == 1:
        return (int(parts[0]) if parts[0] else None, None, 1)
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return (start, end, 1)
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else 1
        return (start, end, step)
    else:
        raise ValueError(f"Invalid frame range format: {range_str}")

def image_provider(start_idx, end_idx, base_url, step=1, buffer_size=60):
    """Downloads images in background thread and yields them sequentially."""
    task_queue = queue.Queue(maxsize=buffer_size)
    session = requests.Session()

    def download_worker(idx):
        url = base_url.format(idx)
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                nparr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return idx, img
            return idx, None
        except Exception:
            return idx, None

    indices = list(range(start_idx, end_idx + 1, step))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(min(len(indices), buffer_size)):
            task_queue.put(executor.submit(download_worker, indices[i]))

        next_to_submit_idx = buffer_size
        
        for i in range(len(indices)):
            future = task_queue.get()
            idx, img = future.result()
            
            yield idx, img

            if next_to_submit_idx < len(indices):
                task_queue.put(executor.submit(download_worker, indices[next_to_submit_idx]))
                next_to_submit_idx += 1

def main(args):
    # Initialize Tracker
    tracker = AthletePositionTracker(
        args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )

    # Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle Frame Range and Step
    start_f, end_f, step_f = 1, 10208, 1
    if args.frame_range:
        s, e, step = parse_frame_range(args.frame_range)
        start_f = s if s is not None else start_f
        end_f = e if e is not None else end_f
        step_f = step

    total_frames = (end_f - start_f) // step_f
    print(f"--- Processing {total_frames} frames ---")

    # Process Loop
    processed_count = 0
    for frame_number, img in image_provider(start_f, end_f, args.base_url, step=step_f):
        if img is None:
            print(f"⚠️  Skip: Frame {frame_number} (Download Failed)")
            continue
        
        frame_name = f"shots_{frame_number:05d}.png"
        
        # Process frame (logs automatically)
        tracker.process_frame(
            img, 
            frame_name, 
            frame_number,
            show_person_bbox=True,
            filter_front_row=True
        )
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f" Processed {processed_count}/{total_frames} frames...")
        
        # Save annotated frame if requested
        if args.save_processed:
            save_path = os.path.join(args.output_dir, frame_name)
            cv2.imwrite(save_path, img)

        if args.debug:
            cv2.imshow("Athlete Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Export Results
    print("\n" + "="*60)
    print(" PROCESSING COMPLETE")
    print("="*60)
    
    tracker.print_summary()
    
    # Export comprehensive frame data (NEW - This is what you'll use!)
    comprehensive_path = os.path.join(args.output_dir, "comprehensive_tracking_data.json")
    tracker.export_comprehensive_data(comprehensive_path)
    
    # Export legacy events
    events_path = os.path.join(args.output_dir, "tracking_events.json")
    tracker.export_events(events_path)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athlete Tracker with Comprehensive Logging")
    parser.add_argument("--base_url", required=True, help="URL with {:05d} placeholder")
    parser.add_argument("--db", required=True, help="Path to face database")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--save_processed", action="store_true", help="Save annotated frames")
    parser.add_argument("--threshold", type=float, default=0.35, help="Face recognition threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="Person detection confidence")
    parser.add_argument("--frame-range", type=str, default=None, help="Format: 'start:end:step'")
    parser.add_argument("--tracker-config", type=str, default=None, help="YOLO tracker config")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")

    args = parser.parse_args()
    main(args)