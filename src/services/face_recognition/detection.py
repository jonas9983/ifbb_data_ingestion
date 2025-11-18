"""
detection.py

Handles face recognition and person detection with bounding box association.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from src.services.face_recognition.depth_detection import DepthAnalyzer

@dataclass
class DetectedAthlete:
    """Represents a detected athlete with face and person information."""
    name: str
    face_bbox: List[int]  # [x1, y1, x2, y2]
    person_bbox: Optional[List[int]]  # [x1, y1, x2, y2], None if not detected
    center_x: float
    face_score: float
    confidence: float = 1.0  # Confidence in the association


class AthleteDetector:
    """
    Detects and recognizes athletes in frames.
    Handles both face recognition and person detection with bbox association.
    """
    
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5):
        """
        Initialize the detector.
        
        Args:
            face_recognizer: FaceRecognizer instance
            person_detector: RF-DETR model instance (optional)
            confidence_threshold: Confidence threshold for person detection
        """
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        
    def detect_faces(self, img) -> List[Dict]:
        """
        Detect and recognize all faces in an image.
        
        Args:
            img: OpenCV image
            
        Returns:
            List of dicts with 'name', 'bbox', 'center_x', 'score'
        """
        faces = self.face_recognizer.app.get(img)
        recognized_faces = []
        
        for face in faces:
            name, score = self.face_recognizer.match(face.embedding)
            
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
        
        return recognized_faces
    
    def detect_persons(self, img) -> List[Dict]:
        """
        Detect person bounding boxes in the image using RF-DETR.
        
        Args:
            img: OpenCV image (BGR format)
            
        Returns:
            List of dicts with 'bbox', 'confidence', 'center_x'
        """
        if self.person_detector is None:
            return []
        
        try:
            # Convert OpenCV BGR to PIL RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_rgb)
            
            # Run RF-DETR prediction
            detections = self.person_detector.predict(
                image_pil, 
                threshold=self.confidence_threshold
            )
            
            # Extract person detections
            persons = []
            
            # Get bounding boxes in xyxy format
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                boxes = detections.xyxy
                confidences = detections.confidence if hasattr(detections, 'confidence') else [1.0] * len(boxes)
                
                for bbox, conf in zip(boxes, confidences):
                    # bbox is expected to be [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    center_x = (x1 + x2) / 2
                    
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'center_x': center_x
                    })
            
            return persons
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []
    
    def associate_face_with_person(
        self, 
        face_bbox: List[int], 
        person_bboxes: List[Dict],
        iou_threshold: float = 0.3
    ) -> Optional[Dict]:
        """
        Associate a face bounding box with a person bounding box.
        
        Args:
            face_bbox: [x1, y1, x2, y2] of face
            person_bboxes: List of person detection dicts
            iou_threshold: Minimum IoU to consider a match
            
        Returns:
            Best matching person bbox dict or None
            
        Strategy: Find person bbox that contains/overlaps with face bbox
        """
        if not person_bboxes:
            return None
        
        best_match = None
        best_iou = iou_threshold
        
        for person in person_bboxes:
            iou = self._calculate_iou(face_bbox, person['bbox'])
            
            # Also check if face is contained within person bbox
            contained = self._is_contained(face_bbox, person['bbox'])
            
            if contained or iou > best_iou:
                best_iou = iou
                best_match = person
        
        return best_match
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_contained(self, face_bbox: List[int], person_bbox: List[int]) -> bool:
        """Check if face bbox is contained within person bbox."""
        x1_f, y1_f, x2_f, y2_f = face_bbox
        x1_p, y1_p, x2_p, y2_p = person_bbox
        
        # Face center should be within person bbox
        center_x = (x1_f + x2_f) / 2
        center_y = (y1_f + y2_f) / 2
        
        return (x1_p <= center_x <= x2_p) and (y1_p <= center_y <= y2_p)
    
    def detect_and_associate(self, img, filter_front_row: bool = False) -> List[DetectedAthlete]:
        """
        Full detection pipeline: detect faces, detect persons, associate them.
        
        Args:
            img: OpenCV image
            filter_front_row: If True, only return front row athletes
            
        Returns:
            List of DetectedAthlete objects
        """
        # Step 1: Detect and recognize faces
        faces = self.detect_faces(img)
        
        # Step 2: Detect persons
        persons = self.detect_persons(img)
        
        # Step 3: Associate faces with person bboxes
        detected_athletes = []
        used_person_indices = set()
        
        for face in faces:
            # Try to find matching person bbox
            best_person = self.associate_face_with_person(
                face['bbox'], 
                persons
            )
            
            if best_person:
                person_bbox = best_person['bbox']
                # Mark this person as used (avoid duplicate associations)
                if best_person in persons:
                    idx = persons.index(best_person)
                    used_person_indices.add(idx)
            else:
                person_bbox = None
            
            athlete = DetectedAthlete(
                name=face['name'],
                face_bbox=face['bbox'].tolist() if hasattr(face['bbox'], 'tolist') else face['bbox'],
                person_bbox=person_bbox,
                center_x=face['center_x'],
                face_score=face['score'],
                confidence=best_person['confidence'] if best_person else 1.0
            )
            
            detected_athletes.append(athlete)
        
        # Step 4: Filter for front row if requested
        if filter_front_row and len(detected_athletes) > 0:
            
            frame_height = img.shape[0]
            analyzer = DepthAnalyzer()
            front_row, back_row = analyzer.filter_front_row_athletes(
                detected_athletes, 
                frame_height,
                verbose=True
            )
            
            if back_row:
                print(f"  → Filtered out {len(back_row)} back row athlete(s): "
                      f"{[a.name for a in back_row]}")
            
            return front_row
        
        return detected_athletes
    
    def draw_annotations(
        self, 
        img, 
        athletes: List[DetectedAthlete],
        show_person_bbox: bool = True
    ):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            img: OpenCV image (modified in-place)
            athletes: List of DetectedAthlete objects
            show_person_bbox: Whether to draw person bboxes
        """
        # Sort athletes by x position (left to right)
        sorted_athletes = sorted(athletes, key=lambda x: x.center_x)
        
        for idx, athlete in enumerate(sorted_athletes):
            position = idx + 1  # 1-indexed position
            
            # Draw person bbox if available (in blue)
            if show_person_bbox and athlete.person_bbox:
                p_bbox = athlete.person_bbox
                cv2.rectangle(
                    img, 
                    (p_bbox[0], p_bbox[1]), 
                    (p_bbox[2], p_bbox[3]), 
                    (255, 0, 0),  # Blue
                    2
                )
            
            # Draw face bbox (in green)
            f_bbox = athlete.face_bbox
            cv2.rectangle(
                img, 
                (f_bbox[0], f_bbox[1]), 
                (f_bbox[2], f_bbox[3]), 
                (0, 255, 0),  # Green
                2
            )
            
            # Draw label with position number
            label = f"#{position}: {athlete.name}"
            
            # Add background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img,
                (f_bbox[0], f_bbox[1] - text_height - 10),
                (f_bbox[0] + text_width, f_bbox[1]),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                img, label, (f_bbox[0], f_bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )