import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.services.athlete_tracking.depth_detection import DepthAnalyzer

@dataclass
class DetectedAthlete:
    name: str
    track_id: Optional[int] 
    person_bbox: List[int]
    center_x: float
    confidence: float = 1.0
    face_bbox: Optional[List[int]] = None 
    is_front_row: bool = True
    debug_face_score: float = 0.0
    debug_raw_face_name: str = "None"

class AthleteDetector:
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5, debug_mode=False):
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        self.depth_analyzer = DepthAnalyzer()
        self.debug_mode = debug_mode
        
        self.athlete_registry: Dict[int, str] = {}
        self.frames_since_check: Dict[int, int] = {}
        self.RECOGNITION_INTERVAL = 15
        self.overwrite_threshold = 0.75

    def detect_faces(self, img) -> List[Dict]:
        faces = self.face_recognizer.app.get(img)
        recognized_faces = []
        for face in faces:
            name, score = self.face_recognizer.match(face.embedding)
            bbox = face.bbox.astype(int)
            recognized_faces.append({
                'name': name, 
                'bbox': bbox.tolist(), 
                'center_x': (bbox[0] + bbox[2]) / 2, 
                'score': score 
            })
        return recognized_faces

    def _match_face_to_person(self, person_bbox, faces_list) -> Optional[Dict]:
        if not faces_list: return None
        px1, py1, px2, py2 = person_bbox
        
        best_face = None
        best_score = 0.0

        for face in faces_list:
            fx1, fy1, fx2, fy2 = face['bbox']
            fcx, fcy = int((fx1+fx2)/2), int((fy1+fy2)/2)
            score = 0.0
            
            # Simple check: Is the center of the face inside the person's bounding box?
            if (px1 <= fcx <= px2) and (py1 <= fcy <= py2):
                score += 1.0
            
            if score > best_score and score > 0.5:
                best_score = score
                best_face = face
        return best_face

    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. Track Persons (No masks anymore)
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        if len(detections) == 0:
            return []
            
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        max_h_ratio = np.max(heights) / img_h
        
        if max_h_ratio < 0.15:
            if self.debug_mode:
                print(f"   [!] Audience/Wide shot detected (Max size: {max_h_ratio:.0%}). Skipping.")
            return []

        detected_athletes = []
        faces = None
        global_faces_checked = False
        used_faces_indices = set()
        
        # 2. Process Each Person
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            x1, y1, x2, y2 = bbox
            
            if y2 > (img_h * 0.95) and (y2 - y1) < (img_h * 0.15): continue

            assigned_name = "Unknown"
            new_face_name, new_face_score, face_bbox = "Unknown", 0.0, None
            
            needs_check = False
            if track_id not in self.athlete_registry:
                needs_check = True
            elif self.frames_since_check.get(track_id, 0) >= self.RECOGNITION_INTERVAL:
                needs_check = True

            current_registry_name = self.athlete_registry.get(track_id, "Unknown")

            if needs_check:
                if not global_faces_checked:
                    faces = self.detect_faces(img)
                    global_faces_checked = True

                avail_faces = [f for idx, f in enumerate(faces) if idx not in used_faces_indices]
                matched_face = self._match_face_to_person(bbox.tolist(), avail_faces)
                self.frames_since_check[track_id] = 0
                
                if matched_face:
                    face_bbox = matched_face['bbox']
                    new_face_name = matched_face['name']
                    new_face_score = matched_face['score']
                    
                    for idx, f in enumerate(faces):
                        if f is matched_face: used_faces_indices.add(idx)
                        
                    if new_face_name != "Unknown":
                        if current_registry_name == "Unknown" or new_face_score > self.overwrite_threshold:
                            
                            for existing_id, existing_name in list(self.athlete_registry.items()):
                                if existing_name == new_face_name and existing_id != track_id:
                                    if self.debug_mode:
                                        print(f" Reassigning {new_face_name} from ID {existing_id} to ID {track_id}")
                                    del self.athlete_registry[existing_id]
                            
                            self.athlete_registry[track_id] = new_face_name
                            assigned_name = new_face_name
                        else:
                            assigned_name = current_registry_name
                    else:
                        assigned_name = current_registry_name
                else:
                    assigned_name = current_registry_name 
            else:
                self.frames_since_check[track_id] += 1
                assigned_name = current_registry_name

            if assigned_name != "Unknown":
                if any(a.name == assigned_name for a in detected_athletes):
                    assigned_name = "Unknown" # Force the weaker clone to be Unknown

            detected_athletes.append(DetectedAthlete(
                name=assigned_name, track_id=track_id, person_bbox=bbox.tolist(),
                center_x=(x1 + x2) / 2, confidence=conf,
                face_bbox=face_bbox, debug_face_score=new_face_score, 
                debug_raw_face_name=new_face_name
            ))
            
        if check_depth and detected_athletes:
            # Depth Analyzer will now just use the bbox coordinates
            front, back = self.depth_analyzer.filter_front_row_athletes(detected_athletes, img_h, verbose=False)
            for a in front: a.is_front_row = True
            for a in back: a.is_front_row = False
            return front + back
            
        return detected_athletes

    def draw_annotations(self, img, athletes, show_person_bbox=True):
        sorted_athletes = sorted(athletes, key=lambda x: x.is_front_row)
        
        for athlete in sorted_athletes:
            color = (0, 255, 0) if athlete.is_front_row else (0, 0, 255)
            row_tag = "[FRONT]" if athlete.is_front_row else "[BACK]"
            
            x1, y1, x2, y2 = athlete.person_bbox
            label = f"ID:{athlete.track_id} {row_tag} {athlete.name}"
            
            if show_person_bbox:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)