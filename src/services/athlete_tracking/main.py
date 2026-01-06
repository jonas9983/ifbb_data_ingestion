import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import json
import argparse
import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import queue
import requests
from src.services.faces.face_recognizer import FaceRecognizer

from src.services.athlete_tracking.detection import AthleteDetector
from src.services.athlete_tracking.swap_detection import SwapDetector
from src.services.athlete_tracking.yolo_segmentation import YOLOSegmentationModel 

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
        
        # Initialize YOLO with optional tracker config
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
        
        self.last_gray_frame: Optional[np.ndarray] = None
        self.cut_threshold: float = 25.0  # Tunable: Average pixel difference threshold (0-255)

    def _check_for_camera_cut(self, img: np.ndarray, frame_name: str) -> bool:
        """Compares current frame to previous frame to detect sudden cuts/fades."""
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.last_gray_frame is None:
            self.last_gray_frame = current_gray
            return False

        # Calculate absolute difference between frames
        diff = cv2.absdiff(current_gray, self.last_gray_frame)
        
        # Calculate the average pixel difference across the image
        avg_diff = np.mean(diff)
        
        is_cut = avg_diff > self.cut_threshold
        
        # Update last frame for the next iteration
        self.last_gray_frame = current_gray
        
        if is_cut:
            print(f"CAMERA CUT: {frame_name}")
            
            if hasattr(self.detector.person_detector.model, 'predictor') and \
            hasattr(self.detector.person_detector.model.predictor, 'trackers'):
                for tracker in self.detector.person_detector.model.predictor.trackers:
                    tracker.reset()
            
            # Clear athlete registry
            self.detector.athlete_registry.clear()
            self.detector.marshall_track_id = -1  # Also reset marshall tracking
            
            return True
        
        return False

    def process_frame(
        self, 
        img, 
        frame_name: str,
        frame_number: int,
        show_person_bbox: bool = True,
        filter_front_row: bool = True
    ):
        
        # --- Camera Cut Detection---
        is_cut = self._check_for_camera_cut(img, frame_name)
        
        if is_cut:
            # 1. Reset the swap detector state immediately
            # This records the 'camera_cut' event and clears last_frame_positions
            self.swap_detector.reset_state(frame_number, frame_name, "camera_cut") 
        
        # 1. Detect ALL athletes (Front and Back)
        all_athletes = self.detector.detect_and_associate(img, check_depth=filter_front_row)
        
        # 2. Filter: We only want to track SWAPS for NAMED front row athletes
        valid_front_row_athletes = [
            a for a in all_athletes 
            if a.is_front_row and a.name != "Unknown" and a.name != "MARSHALL"
            
        ]
        
        # Create a dictionary of recognized athlete positions (Name: Center_X)
        current_positions = {
            athlete.name: athlete.center_x 
            for athlete in valid_front_row_athletes
        }
        
        # 3. Update Swap Detector with clean data
        # Note: If is_cut was True, swap_detector state is reset, and this
        # update_state will now treat this frame as the start of a new scene.
        self.swap_detector.update_state(
            current_positions,
            frame_name,
            frame_number
        )
        
        # 4. Visualize: Draw annotations on the image (using all_athletes for display)
        self.detector.draw_annotations(img, all_athletes, show_person_bbox)
        
        return all_athletes
    
    def export_events(self, output_path: str):
        """Export all recorded events to JSON file."""
        stats = self.swap_detector.get_statistics()
        
        events_data = {
            'summary': stats,
            'events': [event.to_dict() for event in self.swap_detector.events]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Events exported to: {output_path}")
    
    def print_summary(self):
        """Print a summary of tracking results."""
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
    """Downloads images in a background thread and yields them sequentially with step."""
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

    # Generate the list of indices we actually need
    indices = list(range(start_idx, end_idx + 1, step))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Initial pre-fill of the buffer
        for i in range(min(len(indices), buffer_size)):
            task_queue.put(executor.submit(download_worker, indices[i]))

        next_to_submit_idx = buffer_size
        
        for i in range(len(indices)):
            future = task_queue.get()
            idx, img = future.result()
            
            yield idx, img

            # Submit next index in the sequence if available
            if next_to_submit_idx < len(indices):
                task_queue.put(executor.submit(download_worker, indices[next_to_submit_idx]))
                next_to_submit_idx += 1

def main(args):
    # 1. Initialize Tracker
    tracker = AthletePositionTracker(
        args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )

    # 2. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Handle Frame Range and Step
    start_f, end_f, step_f = 1, 10208, 1
    if args.frame_range:
        s, e, step = parse_frame_range(args.frame_range)
        start_f = s if s is not None else start_f
        end_f = e if e is not None else end_f
        step_f = step

    print(f"--- Processing {(end_f - start_f)/step_f} frames ---")

    # 4. Process Loop
    for frame_number, img in image_provider(start_f, end_f, args.base_url, step=step_f):
        if img is None:
            print(f"Skip: Frame {frame_number} (Download Failed)")
            continue
        
        frame_name = f"shots_{frame_number:05d}.png"
        
        # Process frame
        tracker.process_frame(
            img, 
            frame_name, 
            frame_number,
            show_person_bbox=True,
            filter_front_row=True
        )
        
        # 5. Save if requested (Useful for verifying temporal logic)
        if args.save_processed:
            save_path = os.path.join(args.output_dir, frame_name)
            cv2.imwrite(save_path, img)

        if args.debug:
            cv2.imshow("Athlete Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 6. Cleanup & Export
    tracker.print_summary()
    events_path = os.path.join(args.output_dir, "tracking_events.json")
    tracker.export_events(events_path)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athlete Tracker")
    parser.add_argument("--base_url", required=True, help="URL with {:05d}")
    parser.add_argument("--db", required=True, help="Path to face database")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--save_processed", action="store_true", help="Save annotated frames")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--frame-range", type=str, default=None, help="'start:end:step'")
    parser.add_argument("--tracker-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)