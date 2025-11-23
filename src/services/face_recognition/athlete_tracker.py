import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import json
import argparse
from typing import Tuple, Optional
from src.services.face_recognition.face_recognizer import FaceRecognizer

# Import modular components
from detection import AthleteDetector
from swap_detector import SwapDetector
from yolo_segmentation import YOLOSegmentationModel 

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
        
        # Pass debug_mode to Detector
        self.detector = AthleteDetector(
            face_recognizer,
            person_model,
            confidence_threshold=confidence_threshold,
            debug_mode=debug_mode # <--- PASSED DOWN
        )
        self.swap_detector = SwapDetector()
    
    def process_frame(
        self, 
        img, 
        frame_name: str,
        frame_number: int,
        show_person_bbox: bool = True,
        filter_front_row: bool = True
    ):
        
        # 1. Detect ALL athletes
        all_athletes = self.detector.detect_and_associate(img, check_depth=filter_front_row)
        
        # 2. Filter: We only want to track SWAPS for the front row
        front_row_athletes = [a for a in all_athletes if a.is_front_row]
        
        current_positions = {
            athlete.name: athlete.center_x 
            for athlete in front_row_athletes
        }
        
        self.swap_detector.update_state(
            current_positions,
            frame_name,
            frame_number
        )
        
        # 3. Visualize
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
        
        print(f"\n✅ Events exported to: {output_path}")
    
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
    
    # Initialize tracker with debug flag
    tracker = AthletePositionTracker(
        args.db,
        threshold=args.threshold,
        confidence_threshold=args.confidence,
        tracker_config=args.tracker_config,
        debug_mode=args.debug # <--- PASSED HERE
    )
    
    # Setup output directory
    processed_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
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
        
        if img is None:
            continue
        
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
        cv2.imwrite(os.path.join(processed_dir, img_name), img)
    
    # Print summary
    tracker.print_summary()
    
    # Export events to JSON
    events_path = os.path.join(args.data_dir, "processed", "tracking_events.json")
    tracker.export_events(events_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track athlete positions and detect swaps (Refactored)"
    )
    
    parser.add_argument("--data_dir", required=True, help="Directory containing image frames.")
    parser.add_argument("--db", required=True, help="Path to the saved face database (.npz file).")
    parser.add_argument("--threshold", type=float, default=0.35, help="Recognition cosine similarity threshold.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for person detection.")
    parser.add_argument("--frame-range", type=str, default=None, help="Frame range 'start:end:step'")
    parser.add_argument("--tracker-config", type=str, default=None, help="Custom BoT-SORT YAML config (ReID enabled).")
    parser.add_argument("--debug", action="store_true", help="Enable heavy debug logging and visualization.")
    
    args = parser.parse_args()
    main(args)