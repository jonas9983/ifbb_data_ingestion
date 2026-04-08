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
    # DEBUG FIELDS
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
        
        # Identity and Memory Tracking
        self.athlete_registry: Dict[int, str] = {}
        self.frames_since_check: Dict[int, int] = {}
        self.RECOGNITION_INTERVAL = 15  # Only run face recognition every 15 frames per athlete
        self.overwrite_threshold = 0.75
        self.marshall_track_id = -1
        
        # Marshall temporal tracking
        self.marshall_history: List[Dict] = [] 
        self.marshall_history_window = 5
        self.marshall_min_score = 0.6 

    # FACE RECOGNITION (CROPPED)
    def _recognize_face_in_crop(self, img: np.ndarray, bbox: List[int]) -> Tuple[str, float, Optional[List[int]]]:
        """Crops the image to the person's bounding box and runs face recognition."""
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        
        # Add a 20px padding so we don't chop off the top of the head
        pad = 20
        cy1, cy2 = max(0, y1 - pad), min(h, y2 + pad)
        cx1, cx2 = max(0, x1 - pad), min(w, x2 + pad)
        
        crop = img[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            return "Unknown", 0.0, None

        faces = self.face_recognizer.app.get(crop)
        if not faces:
            return "Unknown", 0.0, None

        # Take the most prominent face in this specific crop
        best_face = max(faces, key=lambda f: f.det_score)
        name, score = self.face_recognizer.match(best_face.embedding)
        
        # Convert crop-relative face coordinates back to absolute image coordinates
        fx1, fy1, fx2, fy2 = best_face.bbox.astype(int)
        global_face_bbox = [fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1]
        
        return name, score, global_face_bbox

    # MARSHALL DETECTION
    def _calculate_marshall_score(self, bbox: List[int], mask: Optional[np.ndarray], img: np.ndarray) -> float:
        # [Unchanged from your original code]
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
        if not relevant_scores:
            return False
            
        avg_score = sum(relevant_scores) / len(relevant_scores)
        return len(relevant_scores) >= 3 and avg_score >= self.marshall_min_score

    def _identify_best_marshall(self, detections, img, img_h) -> Optional[Dict]:
        """Loops through detections purely to find the Marshall."""
        candidates = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15): continue # Partial body
            
            score = self._calculate_marshall_score(bbox.tolist(), mask, img)
            candidates.append({'index': i, 'track_id': track_id, 'score': score, 'bbox': bbox, 'mask': mask})
            
        if not candidates: return None
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[0]
        
        if self._determine_marshall_with_temporal_smoothing(top['track_id'], top['score']):
            self.marshall_track_id = top['track_id']
            return top
        return None

    # MAIN ORCHESTRATOR
    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. TRACK PERSONS (Super Fast YOLO)
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        # 2. IDENTIFY MARSHALL
        best_marshall = self._identify_best_marshall(detections, img, img_h)
        
        detected_athletes = []
        
        # 3. PROCESS EACH PERSON
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            # Filter partial bodies at the bottom of the screen
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15):
                continue
                
            # --- HANDLE MARSHALL ---
            if best_marshall and best_marshall['index'] == i:
                if track_id in self.athlete_registry:
                    del self.athlete_registry[track_id] # Clean up
                    
                detected_athletes.append(DetectedAthlete(
                    name="MARSHALL", track_id=track_id, person_bbox=bbox.tolist(),
                    mask=mask, center_x=(bbox[0] + bbox[2]) / 2, confidence=conf,
                    is_front_row=False, debug_marshall_score=best_marshall['score']
                ))
                continue

            # --- HANDLE ATHLETES ---
            assigned_name = "Unknown"
            new_face_name, new_face_score, face_bbox = "Unknown", 0.0, None
            
            # Check if we need to run heavy face recognition
            needs_check = False
            if track_id not in self.athlete_registry:
                needs_check = True
            elif self.frames_since_check.get(track_id, 0) >= self.RECOGNITION_INTERVAL:
                needs_check = True

            current_registry_name = self.athlete_registry.get(track_id, "Unknown")

            if needs_check:
                # RUN RECOGNITION (Only on cropped body!)
                new_face_name, new_face_score, face_bbox = self._recognize_face_in_crop(img, bbox.tolist())
                self.frames_since_check[track_id] = 0
                
                # Logic to update registry
                if new_face_name != "Unknown":
                    if new_face_name == current_registry_name or new_face_score > self.overwrite_threshold:
                        self.athlete_registry[track_id] = new_face_name
                        assigned_name = new_face_name
                    else:
                        assigned_name = current_registry_name # Keep old name if new score is low
                else:
                    assigned_name = current_registry_name # No face found, rely on memory
            else:
                # FAST PATH: Skip face recognition entirely!
                self.frames_since_check[track_id] += 1
                assigned_name = current_registry_name

            detected_athletes.append(DetectedAthlete(
                name=assigned_name, track_id=track_id, person_bbox=bbox.tolist(),
                mask=mask, center_x=(bbox[0] + bbox[2]) / 2, confidence=conf,
                face_bbox=face_bbox, debug_face_score=new_face_score, 
                debug_raw_face_name=new_face_name, debug_marshall_score=0.0
            ))
            
        # 4. DEPTH FILTERING
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
                color = (128, 128, 128)  # Gray
                row_tag = "[MARSHALL]"
            else:
                color = (0, 255, 0) if athlete.is_front_row else (0, 0, 255)
                row_tag = "[FRONT]" if athlete.is_front_row else "[BACK]"
            
            # 1. Draw MASK
            if athlete.mask is not None:
                m = athlete.mask.astype(np.uint8) * 255 if athlete.mask.dtype == bool else athlete.mask
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, -1)
                cv2.drawContours(img, contours, -1, (255,255,255), 1)
            
            # 2. Draw TRACKER Label
            x1, y1, x2, y2 = athlete.person_bbox
            label = f"ID:{athlete.track_id} {row_tag} {athlete.name}"
            
            if self.debug_mode and athlete.debug_marshall_score > 0:
                label += f" [M:{athlete.debug_marshall_score:.2f}]"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            # 3. DEBUG: Draw RAW FACE Detection
            if self.debug_mode and athlete.face_bbox is not None:
                fx1, fy1, fx2, fy2 = athlete.face_bbox
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                raw_info = f"{athlete.debug_raw_face_name} ({athlete.debug_face_score:.2f})"
                cv2.putText(img, raw_info, (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)