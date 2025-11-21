"""
detection.py

Handles face recognition and person detection with instance segmentation association.
Refactored to handle depth flags for visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from src.services.face_recognition.depth_detection import DepthAnalyzer

@dataclass
class DetectedAthlete:
    """Represents a detected athlete with face, person, and depth information."""
    name: str
    face_bbox: List[int]  # [x1, y1, x2, y2]
    person_bbox: Optional[List[int]]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] # Boolean or uint8 mask of the person
    center_x: float
    face_score: float
    confidence: float = 1.0
    is_front_row: bool = True # New flag to track depth state

class AthleteDetector:
    """
    Detects and recognizes athletes in frames.
    Handles face recognition and RF-DETR Instance Segmentation.
    """
    
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5):
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        # Initialize depth analyzer here to keep it ready
        self.depth_analyzer = DepthAnalyzer()
        
    def detect_faces(self, img) -> List[Dict]:
        """Detect and recognize all faces in an image."""
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
        """Detect person masks and bboxes using Segmentation Model."""
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
            
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                boxes = detections.xyxy
                confidences = detections.confidence if hasattr(detections, 'confidence') else [1.0] * len(boxes)
                raw_masks = getattr(detections, 'mask', None)
                
                if raw_masks is None:
                    masks = [None] * len(boxes)
                else:
                    masks = raw_masks

                for i, (bbox, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    center_x = (x1 + x2) / 2
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
            return []
    
    def associate_face_with_person(
        self, 
        face_bbox: List[int], 
        person_data_list: List[Dict],
        iou_threshold: float = 0.3
    ) -> Optional[Dict]:
        """Associate face with person using Mask Check + BBox Fallback."""
        if not person_data_list:
            return None
        
        face_cx = int((face_bbox[0] + face_bbox[2]) / 2)
        face_cy = int((face_bbox[1] + face_bbox[3]) / 2)
        face_y_max = face_bbox[3]

        # --- Mask Point Check ---
        for person in person_data_list:
            mask = person.get('mask')
            if mask is not None:
                h, w = mask.shape[:2]
                if 0 <= face_cx < w and 0 <= face_cy < h:
                    if mask[face_cy, face_cx] > 0:
                        # Face should be in upper portion of body
                        person_bbox = person['bbox']
                        person_height = person_bbox[3] - person_bbox[1]
                        face_relative_y = face_y_max - person_bbox[1]
                        if face_relative_y < (person_height * 0.6):
                            return person

        # --- BBox Fallback ---
        best_match = None
        best_score = 0
        
        for person in person_data_list:
            person_bbox = person['bbox']
            iou = self._calculate_iou(face_bbox, person_bbox)
            contained = self._is_contained(face_bbox, person_bbox)
            
            person_height = person_bbox[3] - person_bbox[1]
            face_relative_y = face_y_max - person_bbox[1]
            vertical_ratio = face_relative_y / person_height if person_height > 0 else 1.0
            
            if contained or iou > iou_threshold:
                score = iou * (1.0 - max(0, vertical_ratio - 0.6))
                if score > best_score:
                    best_score = score
                    best_match = person
        
        return best_match if best_score > 0.1 else None
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i: return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _is_contained(self, face_bbox: List[int], person_bbox: List[int]) -> bool:
        x1_f, y1_f, x2_f, y2_f = face_bbox
        x1_p, y1_p, x2_p, y2_p = person_bbox
        center_x = (x1_f + x2_f) / 2
        center_y = (y1_f + y2_f) / 2
        return (x1_p <= center_x <= x2_p) and (y1_p <= center_y <= y2_p)
    
    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        """
        Full pipeline: Faces -> Persons -> Association -> Depth Check.
        Returns ALL athletes (front and back row).
        """
        # 1. Faces
        faces = self.detect_faces(img)
        
        # 2. Persons
        persons = self.detect_persons(img)
        
        detected_athletes = []
        
        # 3. Association
        for face in faces:
            best_person = self.associate_face_with_person(face['bbox'], persons)
            
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
                mask=person_mask,
                center_x=face['center_x'],
                face_score=face['score'],
                confidence=confidence,
                is_front_row=True # Assume front row initially
            )
            detected_athletes.append(athlete)
        
        # 4. Depth Logic (Updates flags)
        if check_depth and len(detected_athletes) > 0:
            frame_height = img.shape[0]
            
            # Use your existing analyzer logic
            front_row_list, back_row_list = self.depth_analyzer.filter_front_row_athletes(
                detected_athletes, 
                frame_height,
                verbose=True
            )
            
            # Update the flags based on the analyzer's decision
            # (We re-merge into one list to return to the main loop)
            for athlete in front_row_list:
                athlete.is_front_row = True
                
            for athlete in back_row_list:
                athlete.is_front_row = False
            
            # Re-combine list for drawing purposes (order doesn't strictly matter here)
            return front_row_list + back_row_list
        
        return detected_athletes
    
    def draw_annotations(
        self, 
        img, 
        athletes: List[DetectedAthlete],
        show_person_bbox: bool = True
    ):
        """
        Draw bounding boxes and SEGMENTATION MASKS.
        Handles Colors: Green (Front Row), Red (Back Row).
        """
        overlay = img.copy()
        alpha = 0.4
        
        # Sort so front row (is_front_row=True) is drawn LAST (on top)
        sorted_athletes = sorted(athletes, key=lambda x: x.is_front_row)
        
        for idx, athlete in enumerate(sorted_athletes):
            
            # --- Determine Color based on Depth ---
            if athlete.is_front_row:
                color = (0, 255, 0)     # Green
                text_color = (0, 0, 0)  # Black text on Green
            else:
                color = (0, 0, 255)     # Red
                text_color = (255, 255, 255) # White text on Red
            
            # --- Draw Mask Overlay ---
            if athlete.mask is not None:
                mask_uint8 = athlete.mask
                if mask_uint8.dtype == bool:
                    mask_uint8 = mask_uint8.astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1) # Fill
                cv2.drawContours(img, contours, -1, (255, 255, 255), 1) # White border
            
            # --- Draw BBox ---
            f_bbox = athlete.face_bbox
            cv2.rectangle(
                img, 
                (f_bbox[0], f_bbox[1]), 
                (f_bbox[2], f_bbox[3]), 
                color, 
                2
            )
            
            # --- Draw Label ---
            label = f"{athlete.name}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(
                img,
                (f_bbox[0], f_bbox[1] - text_height - 10),
                (f_bbox[0] + text_width, f_bbox[1]),
                color,
                -1
            )
            cv2.putText(
                img, label, (f_bbox[0], f_bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
            )

        # Apply transparency
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)