"""
detection.py
Refactored to use YOLO TRACKING IDs.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.services.face_recognition.depth_detection import DepthAnalyzer

@dataclass
class DetectedAthlete:
    name: str
    track_id: Optional[int] # The YOLO ID (e.g., 1, 2, 5)
    person_bbox: List[int]
    mask: Optional[np.ndarray]
    center_x: float
    confidence: float = 1.0
    face_bbox: Optional[List[int]] = None 
    is_front_row: bool = True 

class AthleteDetector:
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5):
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        self.depth_analyzer = DepthAnalyzer()
        
        # MEMORY: Maps Track ID (int) -> Name (str)
        # Example: {1: "Keone Pearson", 2: "Shaun Clarida"}
        self.athlete_registry: Dict[int, str] = {} 

    def detect_faces(self, img) -> List[Dict]:
        """Standard face detection"""
        faces = self.face_recognizer.app.get(img)
        recognized_faces = []
        for face in faces:
            name, score = self.face_recognizer.match(face.embedding)
            bbox = face.bbox.astype(int)
            recognized_faces.append({
                'name': name, 
                'bbox': bbox, 
                'center_x': (bbox[0] + bbox[2]) / 2, 
                'score': score
            })
        return recognized_faces
    
    def _match_face_to_person(self, person_bbox, person_mask, faces_list) -> Optional[Dict]:
        """Helper to find which face belongs to this body"""
        if not faces_list: return None
        px1, py1, px2, py2 = person_bbox
        
        best_face = None
        best_score = -1.0

        for face in faces_list:
            fx1, fy1, fx2, fy2 = face['bbox']
            fcx, fcy = int((fx1+fx2)/2), int((fy1+fy2)/2)
            score = 0.0
            
            # 1. Mask Check (Precise)
            if person_mask is not None:
                h, w = person_mask.shape[:2]
                if 0 <= fcx < w and 0 <= fcy < h and person_mask[fcy, fcx] > 0:
                    score += 2.0
            
            # 2. BBox Check (Fallback)
            if (px1 <= fcx <= px2) and (py1 <= fcy <= py2):
                score += 1.0
            
            if score > best_score and score > 0.5:
                best_score = score
                best_face = face
        return best_face

    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. TRACK Persons (Get IDs)
        # We use .track() from the updated YOLOSegmentationModel
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        # 2. Detect Faces
        faces = self.detect_faces(img)
        
        detected_athletes = []
        used_faces_indices = set()
        
        # 3. Process each Tracked Person
        # Supervision Detections object allows iteration
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            # --- A. Filter Audience (Bottom 5%) ---
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15):
                continue # Skip audience

            # --- B. Name Resolution Strategy ---
            assigned_name = "Unknown"
            
            # Check Memory First
            if track_id != -1 and track_id in self.athlete_registry:
                assigned_name = self.athlete_registry[track_id]
            
            # Try to find a face to confirm/update name
            # We only look for faces if we don't know the name OR we want to re-verify
            avail_faces = [f for idx, f in enumerate(faces) if idx not in used_faces_indices]
            matched_face = self._match_face_to_person(bbox, mask, avail_faces)
            
            face_bbox = None
            if matched_face:
                face_bbox = matched_face['bbox']
                face_name = matched_face['name']
                
                # Mark face as used
                for idx, f in enumerate(faces):
                    if f is matched_face: used_faces_indices.add(idx)

                # UPDATE MEMORY: If we found a named face, update the registry
                if face_name != "Unknown":
                    self.athlete_registry[track_id] = face_name
                    assigned_name = face_name
            
            # Create Athlete Object
            athlete = DetectedAthlete(
                name=assigned_name,
                track_id=track_id,
                person_bbox=bbox.tolist(),
                mask=mask,
                center_x=(bbox[0] + bbox[2]) / 2,
                confidence=conf,
                face_bbox=face_bbox
            )
            detected_athletes.append(athlete)
        
        # 4. Depth Logic (Front/Back Row)
        if check_depth and detected_athletes:
            front, back = self.depth_analyzer.filter_front_row_athletes(
                detected_athletes, img_h, verbose=False
            )
            for a in front: a.is_front_row = True
            for a in back: a.is_front_row = False
            return front + back
            
        return detected_athletes

    def draw_annotations(self, img, athletes, show_person_bbox=True):
        """Draws Track IDs + Names + Front/Back status"""
        overlay = img.copy()
        alpha = 0.5
        
        sorted_athletes = sorted(athletes, key=lambda x: x.is_front_row)
        
        for athlete in sorted_athletes:
            color = (0, 255, 0) if athlete.is_front_row else (0, 0, 255)
            row_tag = "[FRONT]" if athlete.is_front_row else "[BACK]"
            
            # Mask
            if athlete.mask is not None:
                m = athlete.mask.astype(np.uint8) * 255 if athlete.mask.dtype == bool else athlete.mask
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1)
                cv2.drawContours(img, contours, -1, (255,255,255), 1)
            
            # Label
            x1, y1, x2, y2 = athlete.person_bbox
            label = f"ID:{athlete.track_id} {row_tag} {athlete.name}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)