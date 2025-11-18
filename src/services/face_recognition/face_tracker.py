import os
import cv2
import json
import argparse
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from src.services.face_recognition.face_recognizer import FaceRecognizer


@dataclass
class EventRecord:
    """Represents a tracking event."""
    frame_number: int
    frame_name: str
    timestamp: str
    athletes: List[str]  # Ordered left to right
    trigger: str
    athletes_swapped: Optional[List[Tuple[str, str]]] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert list of tuples to more readable format
        if self.athletes_swapped:
            data['athletes_swapped'] = [
                {'athlete_1': a, 'athlete_2': b} 
                for a, b in self.athletes_swapped
            ]
        return data


class AthletePositionTracker:
    """
    Tracks athlete positions on stage and detects when they swap places.
    Uses face recognition without multi-object tracking (no SORT).
    """
    
    def __init__(self, db_path: str, threshold: float = 0.35):
        """
        Initialize the tracker.
        
        Args:
            db_path: Path to the face database (.npz file)
            threshold: Recognition cosine similarity threshold
        """
        print("Loading FaceRecognizer...")
        self.recognizer = FaceRecognizer(db_path, threshold=threshold)
        print("Recognizer loaded.")
        
        # State tracking
        self.last_frame_positions: Dict[str, float] = {}
        self.all_athletes_seen: Set[str] = set()
        self.frames_with_detection: int = 0
        self.frames_processed: int = 0
        
        # Event recording
        self.events: List[EventRecord] = []
        
    def detect_and_recognize_faces(self, img) -> List[Dict]:
        """
        Detect and recognize all faces in an image.
        
        Args:
            img: OpenCV image
            
        Returns:
            List of dicts with 'name', 'bbox', 'center_x', 'score'
        """
        faces = self.recognizer.app.get(img)
        recognized_faces = []
        
        for face in faces:
            name, score = self.recognizer.match(face.embedding)
            
            # Only track recognized athletes (skip Unknown)
            if name != "Unknown":
                bbox = face.bbox.astype(int)
                center_x = (bbox[0] + bbox[2]) / 2
                
                recognized_faces.append({
                    'name': name,
                    'bbox': bbox,
                    'center_x': center_x,
                    'score': score
                })
                
                # Track new athletes
                if name not in self.all_athletes_seen:
                    self.all_athletes_seen.add(name)
                    print(f"[New Athlete Detected]: {name}")
        
        return recognized_faces
    
    def detect_position_swaps(
        self, 
        current_positions: Dict[str, float], 
        frame_name: str
    ) -> List[Tuple[str, str]]:
        """
        Detect if any athletes swapped positions since last frame.
        
        Args:
            current_positions: Dict mapping athlete name to center_x position
            frame_name: Name of current frame (for logging)
            
        Returns:
            List of tuples (athlete_a, athlete_b) that swapped
        """
        swaps = []
        
        # Need at least 2 athletes in both frames
        if len(self.last_frame_positions) < 2 or len(current_positions) < 2:
            return swaps
        
        # Find athletes present in both frames
        common_athletes = set(self.last_frame_positions.keys()) & set(current_positions.keys())
        
        if len(common_athletes) < 2:
            return swaps
        
        # Check all pairs for position swaps
        common_list = sorted(list(common_athletes))
        
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                athlete_a = common_list[i]
                athlete_b = common_list[j]
                
                # Previous positions
                prev_a = self.last_frame_positions[athlete_a]
                prev_b = self.last_frame_positions[athlete_b]
                
                # Current positions
                curr_a = current_positions[athlete_a]
                curr_b = current_positions[athlete_b]
                
                # Check if relative order changed
                prev_order = "A_left_of_B" if prev_a < prev_b else "B_left_of_A"
                curr_order = "A_left_of_B" if curr_a < curr_b else "B_left_of_A"
                
                if prev_order != curr_order:
                    print("=" * 60)
                    print(f"🎉 POSITION SWAP DETECTED in: {frame_name}")
                    print(f"   Between: {athlete_a} ↔ {athlete_b}")
                    print(f"   Previous: {athlete_a} {'←' if prev_a < prev_b else '→'} {athlete_b}")
                    print(f"   Current:  {athlete_a} {'←' if curr_a < curr_b else '→'} {athlete_b}")
                    print("=" * 60)
                    swaps.append((athlete_a, athlete_b))
        
        return swaps
    
    def draw_annotations(self, img, recognized_faces: List[Dict]):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            img: OpenCV image (modified in-place)
            recognized_faces: List of face data dicts
        """
        # Sort faces by x position (left to right)
        recognized_faces.sort(key=lambda x: x['center_x'])
        
        for idx, face_data in enumerate(recognized_faces):
            bbox = face_data['bbox']
            name = face_data['name']
            position = idx + 1  # 1-indexed position
            
            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label with position number
            label = f"#{position}: {name}"
            
            # Add background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] - text_height - 10),
                (bbox[0] + text_width, bbox[1]),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                img, label, (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
    
    def get_ordered_athletes(self, recognized_faces: List[Dict]) -> List[str]:
        """
        Get list of athlete names ordered left to right.
        
        Args:
            recognized_faces: List of face data dicts
            
        Returns:
            List of athlete names sorted by position
        """
        sorted_faces = sorted(recognized_faces, key=lambda x: x['center_x'])
        return [face['name'] for face in sorted_faces]
    
    def record_event(
        self, 
        frame_number: int,
        frame_name: str,
        athletes: List[str],
        trigger: str,
        swaps: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Record a tracking event.
        
        Args:
            frame_number: Frame number in sequence
            frame_name: Frame filename
            athletes: List of athlete names (left to right)
            trigger: Description of what triggered this event
            swaps: List of (athlete_a, athlete_b) tuples that swapped
        """
        event = EventRecord(
            frame_number=frame_number,
            frame_name=frame_name,
            timestamp=datetime.now().isoformat(),
            athletes=athletes,
            trigger=trigger,
            athletes_swapped=swaps if swaps else None
        )
        self.events.append(event)
    
    def process_frame(
        self, 
        img, 
        frame_name: str,
        frame_number: int
    ) -> List[Dict]:
        """
        Process a single frame: detect, recognize, check swaps, draw.
        
        Args:
            img: OpenCV image
            frame_name: Name of the frame (for logging)
            frame_number: Frame number in sequence
            
        Returns:
            List of recognized faces data
        """
        self.frames_processed += 1
        
        # Detect and recognize faces
        recognized_faces = self.detect_and_recognize_faces(img)
        
        if len(recognized_faces) > 0:
            self.frames_with_detection += 1
        
        # Build current positions dict
        current_positions = {
            face['name']: face['center_x'] 
            for face in recognized_faces
        }
        
        # Get ordered athlete list
        ordered_athletes = self.get_ordered_athletes(recognized_faces)
        
        # Detect swaps
        swaps = self.detect_position_swaps(current_positions, frame_name)
        
        # Record events
        if swaps:
            # Position change event
            self.record_event(
                frame_number=frame_number,
                frame_name=frame_name,
                athletes=ordered_athletes,
                trigger="position_changed",
                swaps=swaps
            )
        elif len(ordered_athletes) > 0:
            # Check if this is a new lineup configuration
            prev_lineup = self.get_current_lineup_names()
            if ordered_athletes != prev_lineup:
                # Athletes appeared/disappeared or changed
                self.record_event(
                    frame_number=frame_number,
                    frame_name=frame_name,
                    athletes=ordered_athletes,
                    trigger="lineup_changed"
                )
        
        # Update state (only if we detected faces)
        if len(current_positions) > 0:
            self.last_frame_positions = current_positions
        
        # Draw annotations
        self.draw_annotations(img, recognized_faces)
        
        return recognized_faces
    
    def get_current_lineup(self) -> List[Tuple[int, str]]:
        """
        Get the current lineup from left to right with positions.
        
        Returns:
            List of (position, name) tuples sorted by position
        """
        sorted_athletes = sorted(
            self.last_frame_positions.items(), 
            key=lambda x: x[1]
        )
        return [(idx + 1, name) for idx, (name, _) in enumerate(sorted_athletes)]
    
    def get_current_lineup_names(self) -> List[str]:
        """
        Get the current lineup as a list of names (left to right).
        
        Returns:
            List of athlete names sorted by position
        """
        sorted_athletes = sorted(
            self.last_frame_positions.items(), 
            key=lambda x: x[1]
        )
        return [name for name, _ in sorted_athletes]
    
    def export_events(self, output_path: str):
        """
        Export all recorded events to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        events_data = {
            'summary': {
                'total_frames_processed': self.frames_processed,
                'frames_with_detection': self.frames_with_detection,
                'total_athletes': len(self.all_athletes_seen),
                'athletes': sorted(list(self.all_athletes_seen))
            },
            'events': [event.to_dict() for event in self.events]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Events exported to: {output_path}")
    
    def print_summary(self):
        """Print a summary of tracking results."""
        print("\n" + "=" * 60)
        print("📊 TRACKING SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {self.frames_processed}")
        print(f"Frames with recognized faces: {self.frames_with_detection}")
        
        if self.frames_processed > 0:
            detection_rate = self.frames_with_detection / self.frames_processed * 100
            print(f"Detection rate: {detection_rate:.1f}%")
        
        print(f"\nTotal unique athletes detected: {len(self.all_athletes_seen)}")
        print("\nAthletes identified:")
        for idx, athlete in enumerate(sorted(self.all_athletes_seen), 1):
            print(f"  {idx}. {athlete}")
        
        if len(self.last_frame_positions) > 0:
            print(f"\nFinal lineup (left to right):")
            for position, name in self.get_current_lineup():
                print(f"  Position {position}: {name}")
        
        print(f"\nTotal events recorded: {len(self.events)}")
        print("=" * 60)


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
    
    # Initialize tracker
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
        tracker.process_frame(img, img_name, frame_number)
        
        # Save processed image
        cv2.imwrite(os.path.join(processed_dir, img_name), img)
    
    # Print summary
    tracker.print_summary()
    
    # Export events to JSON
    events_path = os.path.join(args.data_dir, "tracking_events.json")
    tracker.export_events(events_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track athlete positions and detect swaps (no SORT)"
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
        "--frame_range",
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