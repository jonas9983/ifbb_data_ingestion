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
    mask: Optional[np.ndarray]
    center_x: float
    confidence: float = 1.0
    face_bbox: Optional[List[int]] = None 
    is_front_row: bool = True
    debug_face_score: float = 0.0
    debug_raw_face_name: str = "None"
    debug_marshall_score: float = 0.0

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
        self.marshall_track_id = -1
        
        self.marshall_history: List[Dict] = [] 
        self.marshall_history_window = 5
        self.marshall_min_score = 0.6 

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

    def _match_face_to_person(self, person_bbox, person_mask, faces_list) -> Optional[Dict]:
        if not faces_list: return None
        px1, py1, px2, py2 = person_bbox
        
        crop = img[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            return "Unknown", 0.0, None

        for face in faces_list:
            fx1, fy1, fx2, fy2 = face['bbox']
            fcx, fcy = int((fx1+fx2)/2), int((fy1+fy2)/2)
            score = 0.0
            
            if person_mask is not None:
                h, w = person_mask.shape[:2]
                if 0 <= fcx < w and 0 <= fcy < h and person_mask[fcy, fcx] > 0:
                    score += 2.0
            
            if (px1 <= fcx <= px2) and (py1 <= fcy <= py2):
                score += 1.0
            
            if score > best_score and score > 0.5:
                best_score = score
                best_face = face
        return best_face

    def _calculate_marshall_score(self, bbox: List[int], mask: Optional[np.ndarray], img: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return 0.0
        bbox_h, bbox_w = y2 - y1, x2 - x1
        if bbox_h < 50: return 0.0

        patch = img[y1:y2, x1:x2].copy()
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            mask_bool = mask_crop > 0
            patch[~mask_bool] = 0

        p_h, p_w = patch.shape[:2]
        roi = patch[int(p_h*0.15):int(p_h*0.85), int(p_w*0.30):int(p_w*0.70)]
        if roi.size == 0: return 0.0

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        MIN_BRIGHTNESS = 100
        lower1 = np.array([0, 40, MIN_BRIGHTNESS]) 
        upper1 = np.array([25, 255, 255])
        lower2 = np.array([160, 40, MIN_BRIGHTNESS])
        upper2 = np.array([180, 255, 255])
        
        skin_mask = cv2.inRange(hsv_roi, lower1, upper1) + cv2.inRange(hsv_roi, lower2, upper2)
        skin_pixels = cv2.countNonZero(skin_mask)
        
        if mask is not None:
            v_channel = hsv_roi[:, :, 2]
            person_pixels = cv2.countNonZero(v_channel)
            total_pixels = person_pixels if person_pixels > 0 else 1
        else:
            total_pixels = roi.shape[0] * roi.shape[1]
            
        skin_ratio = skin_pixels / total_pixels
        black_mask = hsv_roi[:, :, 2] < 50
        black_pixels = np.sum(black_mask)
        black_ratio = black_pixels / total_pixels
        
        skin_score = 1.0 - min(skin_ratio / 0.3, 1.0)
        black_score = min(black_ratio / 0.5, 1.0)
        return (skin_score * 0.6) + (black_score * 0.4)

    def _determine_marshall_with_temporal_smoothing(self, track_id: int, current_score: float) -> bool:
        self.marshall_history.append({'track_id': track_id, 'score': current_score})
        if len(self.marshall_history) > self.marshall_history_window:
            self.marshall_history.pop(0)
            
        relevant_scores = [h['score'] for h in self.marshall_history if h['track_id'] == track_id]
        if not relevant_scores: return False
        avg_score = sum(relevant_scores) / len(relevant_scores)
        return len(relevant_scores) >= 3 and avg_score >= self.marshall_min_score

    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. Track Persons
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        # --- STAGE HEURISTIC ---
        if len(detections) == 0:
            return []
            
        # Check the height of the tallest person detected
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        max_h_ratio = np.max(heights) / img_h
        
        # If the tallest person takes up less than 15% of the screen height, it's a crowd/wide shot
        if max_h_ratio < 0.15:
            if self.debug_mode:
                print(f"   [!] Audience/Wide shot detected (Max size: {max_h_ratio:.0%}). Skipping.")
            return []
        
        # 2. Marshall Detection
        marshall_candidates = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15): continue
            
            score = self._calculate_marshall_score(bbox.tolist(), mask, img)
            marshall_candidates.append({'index': i, 'track_id': track_id, 'score': score, 'bbox': bbox, 'mask': mask})
            
        best_marshall = None
        if marshall_candidates:
            marshall_candidates.sort(key=lambda x: x['score'], reverse=True)
            top = marshall_candidates[0]
            if self._determine_marshall_with_temporal_smoothing(top['track_id'], top['score']):
                best_marshall = top
                self.marshall_track_id = top['track_id']

        detected_athletes = []
        
        faces = None
        global_faces_checked = False
        
        # 3. Process Each Person
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            x1, y1, x2, y2 = bbox
            
            if y2 > (img_h * 0.95) and (y2 - y1) < (img_h * 0.15): continue
                
            is_marshall = (best_marshall is not None and best_marshall['index'] == i)
            marshall_score = next((c['score'] for c in marshall_candidates if c['index'] == i), 0.0)

            if is_marshall:
                if track_id in self.athlete_registry:
                    del self.athlete_registry[track_id]
                detected_athletes.append(DetectedAthlete(
                    name="MARSHALL", track_id=track_id, person_bbox=bbox.tolist(),
                    mask=mask, center_x=(x1 + x2) / 2, confidence=conf,
                    is_front_row=False, debug_marshall_score=marshall_score
                ))
                continue

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
                matched_face = self._match_face_to_person(bbox.tolist(), mask, avail_faces)
                self.frames_since_check[track_id] = 0
                
                if matched_face:
                    face_bbox = matched_face['bbox']
                    new_face_name = matched_face['name']
                    new_face_score = matched_face['score']
                    
                    for idx, f in enumerate(faces):
                        if f is matched_face: used_faces_indices.add(idx)
                        
                    if new_face_name != "Unknown":
                        if current_registry_name == "Unknown":
                            self.athlete_registry[track_id] = new_face_name
                            assigned_name = new_face_name
                        elif new_face_name == current_registry_name:
                            assigned_name = current_registry_name
                        elif new_face_score > self.overwrite_threshold:
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

            detected_athletes.append(DetectedAthlete(
                name=assigned_name, track_id=track_id, person_bbox=bbox.tolist(),
                mask=mask, center_x=(x1 + x2) / 2, confidence=conf,
                face_bbox=face_bbox, debug_face_score=new_face_score, 
                debug_raw_face_name=new_face_name, debug_marshall_score=marshall_score
            ))
            
        if check_depth and detected_athletes:
            front, back = self.depth_analyzer.filter_front_row_athletes(detected_athletes, img_h, verbose=False)
            for a in front: a.is_front_row = True
            for a in back: a.is_front_row = False
            return front + back
            
        return detected_athletes

    def draw_annotations(self, img, athletes, show_person_bbox=True):
        overlay = img.copy()
        alpha = 0.5
        sorted_athletes = sorted(athletes, key=lambda x: x.is_front_row)
        
        for athlete in sorted_athletes:
            if athlete.name == "MARSHALL":
                color = (128, 128, 128)
                row_tag = "[MARSHALL]"
            else:
                color = (0, 255, 0) if athlete.is_front_row else (0, 0, 255)
                row_tag = "[FRONT]" if athlete.is_front_row else "[BACK]"
            
            if athlete.mask is not None:
                m = athlete.mask.astype(np.uint8) * 255 if athlete.mask.dtype == bool else athlete.mask
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1)
                cv2.drawContours(img, contours, -1, (255,255,255), 1)
            
            x1, y1, x2, y2 = athlete.person_bbox
            label = f"ID:{athlete.track_id} {row_tag} {athlete.name}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)