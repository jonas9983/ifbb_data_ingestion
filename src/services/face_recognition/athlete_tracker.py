"""
athlete_position_tracker.py (REFACTORED)

Main script that orchestrates detection and swap tracking.
Uses modular components from detection.py and swap_detector.py
"""

import os
import cv2
import json
import argparse
from typing import Tuple, Optional
from rfdetr import RFDETRMedium

from src.services.face_recognition.face_recognizer import FaceRecognizer
from src.services.face_recognition.detection import AthleteDetector
from src.services.face_recognition.swap_detector import SwapDetector


class AthletePositionTracker:
    """
    Main tracker that orchestrates detection and swap detection.
    Refactored to use modular components.
    """
    
    def __init__(self, db_path: str, threshold: float = 0.35, confidence_threshold: float = 0.5):
        """
        Initialize the tracker.
        
        Args:
            db_path: Path to the face database (.npz file)
            threshold: Recognition cosine similarity threshold
            confidence_threshold: Confidence threshold for person detection
        """
        print("Loading FaceRecognizer...")
        face_recognizer = FaceRecognizer(db_path, threshold=threshold)
        print("Recognizer loaded.")
        
        print("Loading RF-DETR person detector...")
        person_model = RFDETRMedium()
        person_model.optimize_for_inference()
        
        # Initialize modular components
        self.detector = AthleteDetector(
            face_recognizer, 
            person_model, 
            confidence_threshold=confidence_threshold
        )
        self.swap_detector = SwapDetector()
    
    def process_frame(
        self, 
        img, 
        frame_name: str,
        frame_number: int,
        show_person_bbox: bool = True
    ):
        """
        Process a single frame: detect, recognize, check swaps, draw.
        
        Args:
            img: OpenCV image
            frame_name: Name of the frame (for logging)
            frame_number: Frame number in sequence
            show_person_bbox: Whether to draw person bounding boxes
            
        Returns:
            List of DetectedAthlete objects
        """
        # Step 1: Detect and recognize athletes
        detected_athletes = self.detector.detect_and_associate(img)
        
        # Step 2: Build positions dict and check for swaps
        current_positions = {
            athlete.name: athlete.center_x 
            for athlete in detected_athletes
        }
        
        # Step 3: Update swap detector state
        self.swap_detector.update_state(
            current_positions,
            frame_name,
            frame_number
        )
        
        # Step 4: Draw annotations
        self.detector.draw_annotations(img, detected_athletes, show_person_bbox)
        
        return detected_athletes
    
    def export_events(self, output_path: str):
        """
        Export all recorded events to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
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
    """
    Parse frame range string in format 'start:end:step'.
    
    Args:
        range_str: String like '0:100:5' or '::10' or '50:150'
        
    Returns:
        Tuple of (start, end, step) where None means use default
        
    Examples:
        '0:100:5' -> (0, 100, 5)
        '::10' -> (None, None, 10)
        '50:150' -> (50, 150, 1)
        '100' -> (100, None, 1)
    """
    parts = range_str.split(':')
    
    if len(parts) == 1:
        # Single number means start from that frame
        return (int(parts[0]) if parts[0] else None, None, 1)
    elif len(parts) == 2:
        # start:end
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return (start, end, 1)
    elif len(parts) == 3:
        # start:end:step
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else 1
        return (start, end, step)
    else:
        raise ValueError(f"Invalid frame range format: {range_str}")


def main(args):
    """Main function to process all frames in a directory."""
    
    # Initialize tracker (person_detector=None for now)
    tracker = AthletePositionTracker(args.db, threshold=args.threshold)
    
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
            show_person_bbox=True  # Set to False to hide person bboxes
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
    
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Directory containing image frames."
    )
    parser.add_argument(
        "--db", 
        required=True, 
        help="Path to the saved face database (.npz file)."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.35, 
        help="Recognition cosine similarity threshold."
    )
    parser.add_argument(
        "--frame-range",
        type=str,
        default=None,
        help=(
            "Frame range to process in format 'start:end:step'. "
            "Examples: '0:100:5' (frames 0-100, every 5th), "
            "'::10' (all frames, every 10th), "
            "'50:150' (frames 50-150, every frame), "
            "'100' (start from frame 100 to end)"
        )
    )
    
    args = parser.parse_args()
    main(args)