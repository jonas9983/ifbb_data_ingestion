import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import json
import argparse
import numpy as np
from typing import Tuple, Optional
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


def main(args):
    """Main function to process all frames in a directory."""
    
    # Initialize tracker with arguments
    tracker = AthletePositionTracker(
        args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug
    )
    
    # Setup output directory
    output_dir = os.path.join(args.data_dir, args.processed_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    all_images = sorted(
        [f for f in os.listdir(args.data_dir) 
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    
    # Parse frame range if provided
    if args.frame_range:
        start, end, step = parse_frame_range(args.frame_range)
        start = start if start is not None else 0
        end = end if end is not None else len(all_images)
        images = all_images[start:end:step]
        print(f"Processing frames {start} to {end} with step {step}")
        print(f"Total frames to process: {len(images)} (out of {len(all_images)} total)")
    else:
        images = all_images
        print(f"Processing all {len(images)} frames")
    
    print("=" * 60)
    
    # Process each frame
    for idx, img_name in enumerate(images):
        img_path = os.path.join(args.data_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # Calculate actual frame number in original sequence
        frame_number = all_images.index(img_name)
        
        # Process frame
        tracker.process_frame(
            img, 
            img_name, 
            frame_number,
            show_person_bbox=True,
            filter_front_row=True
        )
        
        # Save processed image
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    
    # Print summary
    tracker.print_summary()
    
    # Export events to JSON
    events_path = os.path.join(args.data_dir, args.processed_dir, "tracking_events.json")
    tracker.export_events(events_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track athlete positions and detect swaps (Refactored)"
    )
    
    parser.add_argument("--data_dir", required=True, help="Directory containing image frames.")
    parser.add_argument("--db", required=True, help="Path to the saved face database (.npz file).")
    parser.add_argument("--processed_dir", required = False, type = str, default= "processed", help="Define where the processed images will be saved")
    parser.add_argument("--threshold", type=float, default=0.35, help="Recognition cosine similarity threshold.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for person detection.")
    parser.add_argument("--frame-range", type=str, default=None, help="Frame range 'start:end:step'")
    parser.add_argument("--tracker-config", type=str, default=None, help="Custom BoT-SORT YAML config.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and visualization.")

    args = parser.parse_args()
    main(args)