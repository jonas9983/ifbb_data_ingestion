"""
detection.py

Handles face recognition and person detection with instance segmentation association.
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
    person_bbox: Optional[List[int]]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] # Boolean or uint8 mask of the person
    center_x: float
    face_score: float
    confidence: float = 1.0


class AthleteDetector:
    """
    Detects and recognizes athletes in frames.
    Handles face recognition and RF-DETR Instance Segmentation.
    """
    
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5):
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
        Detect person masks and bboxes using RF-DETR Segmentation.
        """
        if self.person_detector is None:
            return []
        
        try:
            # Convert OpenCV BGR to PIL RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_rgb)
            
            detections = self.person_detector.predict(
                image_pil, 
                threshold=self.confidence_threshold
            )
            
            persons = []
            
            # Check if we have detections
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                boxes = detections.xyxy
                # Handle confidences safely
                confidences = detections.confidence if hasattr(detections, 'confidence') else [1.0] * len(boxes)
                
                raw_masks = getattr(detections, 'mask', None)
                
                if raw_masks is not None:
                    masks = raw_masks
                else:
                    # Fallback: detected a person, but no mask found (or model is box-only)
                    masks = [None] * len(boxes)
                    print("Warning: No masks returned by model. Is this a segmentation model?")

                for i, (bbox, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    center_x = (x1 + x2) / 2
                    
                    # Safely get the mask for this index
                    person_mask = masks[i] if i < len(masks) else None

                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'mask': person_mask,
                        'confidence': float(conf),
                        'center_x': center_x
                    })
            
            return persons
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            import traceback
            traceback.print_exc() # Print full error to see exactly where it fails
            return []
    
    def associate_face_with_person(
        self, 
        face_bbox: List[int], 
        person_data_list: List[Dict],
        iou_threshold: float = 0.3
    ) -> Optional[Dict]:
        """
        Associate face with person.
        1. Check if face center is inside the Person Mask (Precision).
        2. Fallback to BBox IoU/Containment if mask check fails.
        """
        if not person_data_list:
            return None
        
        # Calculate face center
        face_cx = int((face_bbox[0] + face_bbox[2]) / 2)
        face_cy = int((face_bbox[1] + face_bbox[3]) / 2)

        # -- Mask Point Check ---
        for person in person_data_list:
            mask = person.get('mask')
            if mask is not None:
                # Ensure coordinates are within bounds
                h, w = mask.shape[:2]
                if 0 <= face_cx < w and 0 <= face_cy < h:
                    # Check if the pixel at face center is part of the mask (True/255)
                    if mask[face_cy, face_cx] > 0:
                        return person

        # --- BBox Fallback (Original Logic) ---
        best_match = None
        best_iou = iou_threshold
        
        for person in person_data_list:
            iou = self._calculate_iou(face_bbox, person['bbox'])
            contained = self._is_contained(face_bbox, person['bbox'])
            
            if contained or iou > best_iou:
                best_iou = iou
                best_match = person
        
        return best_match
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """ Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_contained(self, face_bbox: List[int], person_bbox: List[int]) -> bool:
        """ Check if face bbox is contained within person bbox."""
        x1_f, y1_f, x2_f, y2_f = face_bbox
        x1_p, y1_p, x2_p, y2_p = person_bbox
        center_x = (x1_f + x2_f) / 2
        center_y = (y1_f + y2_f) / 2
        return (x1_p <= center_x <= x2_p) and (y1_p <= center_y <= y2_p)
    
    def detect_and_associate(self, img, filter_front_row: bool = False) -> List[DetectedAthlete]:
        """
        Full detection pipeline.
        """
        # 1. Faces
        faces = self.detect_faces(img)
        
        # 2. Persons (Masks + Boxes)
        persons = self.detect_persons(img)
        
        detected_athletes = []
        
        for face in faces:
            best_person = self.associate_face_with_person(
                face['bbox'], 
                persons
            )
            
            person_bbox = None
            person_mask = None
            confidence = 1.0

            if best_person:
                person_bbox = best_person['bbox']
                person_mask = best_person['mask']
                confidence = best_person['confidence']
            
            athlete = DetectedAthlete(
                name=face['name'],
                face_bbox=face['bbox'].tolist() if hasattr(face['bbox'], 'tolist') else face['bbox'],
                person_bbox=person_bbox,
                mask=person_mask,  # Store the mask
                center_x=face['center_x'],
                face_score=face['score'],
                confidence=confidence
            )
            
            detected_athletes.append(athlete)
        
        # Filter logic handles the DepthAnalyzer
        if filter_front_row and len(detected_athletes) > 0:
            frame_height = img.shape[0]
            analyzer = DepthAnalyzer()
            front_row, _ = analyzer.filter_front_row_athletes(
                detected_athletes, 
                frame_height,
                verbose=True
            )
            return front_row
        
        return detected_athletes
    
    def draw_annotations(
        self, 
        img, 
        athletes: List[DetectedAthlete],
        show_person_bbox: bool = True
    ):
        """
        Draw bounding boxes and SEGMENTATION MASKS on the image.
        """
        # Create an overlay for transparency
        overlay = img.copy()
        alpha = 0.4  # Transparency factor
        
        sorted_athletes = sorted(athletes, key=lambda x: x.center_x)
        
        for idx, athlete in enumerate(sorted_athletes):
            position = idx + 1
            
            # --- Draw Mask Overlay ---
            if athlete.mask is not None:
                # Define colors (Blue-ish for masks)
                color = (255, 100, 0) # BGR: Blue
                
                # If mask is boolean, convert to uint8
                mask_uint8 = athlete.mask
                if mask_uint8.dtype == bool:
                    mask_uint8 = mask_uint8.astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1) # Fill
                cv2.drawContours(img, contours, -1, (255, 255, 255), 2) # White border on main img
            
            # Draw person bbox if requested (optional now that we have masks)
            if show_person_bbox and athlete.person_bbox:
                p_bbox = athlete.person_bbox
                cv2.rectangle(
                    img, 
                    (p_bbox[0], p_bbox[1]), 
                    (p_bbox[2], p_bbox[3]), 
                    (255, 100, 0), 
                    1
                )
            
            # Draw face bbox (Green)
            f_bbox = athlete.face_bbox
            cv2.rectangle(
                img, 
                (f_bbox[0], f_bbox[1]), 
                (f_bbox[2], f_bbox[3]), 
                (0, 255, 0), 
                2
            )
            
            # Label
            label = f"#{position}: {athlete.name}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(
                img,
                (f_bbox[0], f_bbox[1] - text_height - 10),
                (f_bbox[0] + text_width, f_bbox[1]),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                img, label, (f_bbox[0], f_bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        # Apply the transparent overlay
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)