"""
detection.py
Refactored with 'Sticky Identity' logic + Optional Debugging.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.services.face_recognition.depth_detection import DepthAnalyzer

@dataclass
class DetectedAthlete:
    name: str
    track_id: Optional[int] 
    person_bbox: List[int]
    mask: Optional[np.ndarray]
    center_x: float
    confidence: float = 1.0
    face_bbox: Optional[List[int]] = None 
    is_front_row: bool = True
    # DEBUG FIELDS
    debug_face_score: float = 0.0
    debug_raw_face_name: str = "None" 

class AthleteDetector:
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5, debug_mode=False):
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        self.depth_analyzer = DepthAnalyzer()
        self.debug_mode = debug_mode  # <--- NEW FLAG
        
        self.athlete_registry: Dict[int, str] = {} 
        self.overwrite_threshold = 0.75

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
        if not faces_list: return None
        px1, py1, px2, py2 = person_bbox
        
        best_face = None
        best_score = -1.0

        for face in faces_list:
            fx1, fy1, fx2, fy2 = face['bbox']
            fcx, fcy = int((fx1+fx2)/2), int((fy1+fy2)/2)
            score = 0.0
            
            # 1. Mask Check
            if person_mask is not None:
                h, w = person_mask.shape[:2]
                if 0 <= fcx < w and 0 <= fcy < h and person_mask[fcy, fcx] > 0:
                    score += 2.0
            
            # 2. BBox Check
            if (px1 <= fcx <= px2) and (py1 <= fcy <= py2):
                score += 1.0
            
            if score > best_score and score > 0.5:
                best_score = score
                best_face = face
        return best_face

    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. TRACK Persons
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        # 2. Detect Faces
        faces = self.detect_faces(img)
        
        # DEBUG LOGGING
        if self.debug_mode and len(faces) > 0:
            print(f"\n--- Faces Detected: {len(faces)} ---")
            for f in faces:
                print(f"   > Raw Face: {f['name']} (Score: {f['score']:.4f})")

        detected_athletes = []
        used_faces_indices = set()
        
        # 3. Process each Tracked Person
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15):
                continue

            assigned_name = "Unknown"
            
            # Match Face
            avail_faces = [f for idx, f in enumerate(faces) if idx not in used_faces_indices]
            matched_face = self._match_face_to_person(bbox, mask, avail_faces)
            
            face_bbox = None
            new_face_name = "Unknown"
            new_face_score = 0.0

            if matched_face:
                face_bbox = matched_face['bbox']
                new_face_name = matched_face['name']
                new_face_score = matched_face['score']
                
                for idx, f in enumerate(faces):
                    if f is matched_face: used_faces_indices.add(idx)

            # --- LOGIC TRACE ---
            # We build the debug string but only print it if debug_mode is True
            debug_log = f"[Track {track_id}] "
            
            # Case A: Existing History
            if track_id != -1 and track_id in self.athlete_registry:
                current_registry_name = self.athlete_registry[track_id]
                debug_log += f"Mem: '{current_registry_name}'. "
                
                if new_face_name != "Unknown":
                    debug_log += f"Face found: '{new_face_name}' ({new_face_score:.2f}). "
                    if new_face_name == current_registry_name:
                        assigned_name = current_registry_name
                        debug_log += "MATCH -> Confirmed."
                    else:
                        # CONFLICT
                        if new_face_score > self.overwrite_threshold:
                            self.athlete_registry[track_id] = new_face_name
                            assigned_name = new_face_name
                            debug_log += f"CONFLICT -> OVERWRITE (Score > {self.overwrite_threshold})."
                        else:
                            assigned_name = current_registry_name
                            debug_log += f"CONFLICT -> IGNORED (Score {new_face_score:.2f} too low). STICKY HOLD."
                else:
                    assigned_name = current_registry_name
                    debug_log += "No Face -> Using Memory."

            # Case B: New Track
            elif track_id != -1 and new_face_name != "Unknown":
                self.athlete_registry[track_id] = new_face_name
                assigned_name = new_face_name
                debug_log += f"New Track -> Assigned '{new_face_name}'."
            else:
                debug_log += "Unknown."

            # ONLY PRINT IF DEBUG MODE IS ON
            if self.debug_mode:
                print(debug_log)

            athlete = DetectedAthlete(
                name=assigned_name,
                track_id=track_id,
                person_bbox=bbox.tolist(),
                mask=mask,
                center_x=(bbox[0] + bbox[2]) / 2,
                confidence=conf,
                face_bbox=face_bbox,
                debug_face_score=new_face_score,
                debug_raw_face_name=new_face_name
            )
            detected_athletes.append(athlete)
        
        if check_depth and detected_athletes:
            front, back = self.depth_analyzer.filter_front_row_athletes(
                detected_athletes, img_h, verbose=False
            )
            for a in front: a.is_front_row = True
            for a in back: a.is_front_row = False
            return front + back
            
        return detected_athletes

    def draw_annotations(self, img, athletes, show_person_bbox=True):
        overlay = img.copy()
        alpha = 0.5
        
        sorted_athletes = sorted(athletes, key=lambda x: x.is_front_row)
        
        for athlete in sorted_athletes:
            color = (0, 255, 0) if athlete.is_front_row else (0, 0, 255)
            row_tag = "[FRONT]" if athlete.is_front_row else "[BACK]"
            
            # 1. Draw MASK
            if athlete.mask is not None:
                m = athlete.mask.astype(np.uint8) * 255 if athlete.mask.dtype == bool else athlete.mask
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1)
                cv2.drawContours(img, contours, -1, (255,255,255), 1)
            
            # 2. Draw TRACKER Label (The Final Decision)
            x1, y1, x2, y2 = athlete.person_bbox
            label = f"ID:{athlete.track_id} {row_tag} {athlete.name}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            # 3. DEBUG: Draw RAW FACE Detection (Cyan) - ONLY IN DEBUG MODE
            if self.debug_mode and athlete.face_bbox is not None:
                fx1, fy1, fx2, fy2 = athlete.face_bbox
                # Cyan Box for Face
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                
                # Debug Text: "RawName (Score)"
                raw_info = f"{athlete.debug_raw_face_name} ({athlete.debug_face_score:.2f})"
                cv2.putText(img, raw_info, (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)