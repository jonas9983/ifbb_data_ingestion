"""
detection.py - FIXED Marshall Detection
Key changes:
1. Calculate "marshall score" for each person
2. Only assign ONE person as Marshall per frame (highest score)
3. Add temporal smoothing to prevent flickering
"""

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
    debug_marshall_score: float = 0.0  # NEW: For debugging

class AthleteDetector:
    def __init__(self, face_recognizer, person_detector=None, confidence_threshold=0.5, debug_mode=False):
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.confidence_threshold = confidence_threshold
        self.depth_analyzer = DepthAnalyzer()
        self.debug_mode = debug_mode
        
        self.athlete_registry: Dict[int, str] = {} 
        self.overwrite_threshold = 0.75
        self.marshall_track_id = -1
        
        # NEW: Marshall temporal tracking
        self.marshall_history: List[Dict] = []  # Track last N frames
        self.marshall_history_window = 5
        self.marshall_min_score = 0.6  # Minimum score to be considered Marshall

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
    
    def _calculate_marshall_score(self, bbox: List[int], mask: Optional[np.ndarray], img: np.ndarray) -> float:
        """
        Calculate a confidence score (0.0 to 1.0) for how likely this person is the Marshall.
        Higher score = more likely to be Marshall.
        """
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        
        # Validations
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return 0.0
        
        bbox_h, bbox_w = y2 - y1, x2 - x1
        if bbox_h < 50: return 0.0

        # 1. Prepare Image (Masking is Crucial)
        patch = img[y1:y2, x1:x2].copy()
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            mask_bool = mask_crop > 0
            patch[~mask_bool] = 0

        # 2. Define Central Column (Torso area)
        p_h, p_w = patch.shape[:2]
        roi = patch[int(p_h*0.15):int(p_h*0.85), int(p_w*0.30):int(p_w*0.70)]
        if roi.size == 0: return 0.0

        # 3. Calculate Skin Score (LOWER is better for Marshall)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        MIN_BRIGHTNESS = 100
        lower1 = np.array([0, 40, MIN_BRIGHTNESS]) 
        upper1 = np.array([25, 255, 255])
        lower2 = np.array([160, 40, MIN_BRIGHTNESS])
        upper2 = np.array([180, 255, 255])
        
        skin_mask = cv2.inRange(hsv_roi, lower1, upper1) + cv2.inRange(hsv_roi, lower2, upper2)
        skin_pixels = cv2.countNonZero(skin_mask)
        
        # Calculate ratio against NON-BLACK pixels only
        if mask is not None:
            v_channel = hsv_roi[:, :, 2]
            person_pixels = cv2.countNonZero(v_channel)
            total_pixels = person_pixels if person_pixels > 0 else 1
        else:
            total_pixels = roi.shape[0] * roi.shape[1]
            
        skin_ratio = skin_pixels / total_pixels
        
        # 4. Calculate Black Clothing Score (HIGHER is better for Marshall)
        black_mask = hsv_roi[:, :, 2] < 50  # Dark pixels
        black_pixels = np.sum(black_mask)
        black_ratio = black_pixels / total_pixels
        
        # 5. Combined Score (0.0 to 1.0)
        # Marshall should have: LOW skin + HIGH black
        skin_score = 1.0 - min(skin_ratio / 0.3, 1.0)  # Normalize: 0.3 skin = 0 score
        black_score = min(black_ratio / 0.5, 1.0)      # Normalize: 0.5 black = 1.0 score
        
        combined_score = (skin_score * 0.6) + (black_score * 0.4)  # Weighted average
        
        return combined_score

    def _determine_marshall_with_temporal_smoothing(
        self, 
        track_id: int, 
        current_marshall_score: float
    ) -> bool:
        """
        Use temporal smoothing to determine if this track_id is the Marshall.
        Requires consistent high scores over multiple frames.
        """
        # Update history
        self.marshall_history.append({
            'track_id': track_id,
            'score': current_marshall_score
        })
        
        # Keep only recent history
        if len(self.marshall_history) > self.marshall_history_window:
            self.marshall_history.pop(0)
        
        # Calculate average score for this track_id over recent frames
        relevant_scores = [
            h['score'] for h in self.marshall_history 
            if h['track_id'] == track_id
        ]
        
        if not relevant_scores:
            return False
        
        avg_score = sum(relevant_scores) / len(relevant_scores)
        
        # Require consistent high score (e.g., average > 0.6 over 3+ frames)
        if len(relevant_scores) >= 3 and avg_score >= self.marshall_min_score:
            return True
        
        return False

    def detect_and_associate(self, img, check_depth: bool = False) -> List[DetectedAthlete]:
        img_h, img_w = img.shape[:2]
        
        # 1. TRACK Persons
        detections = self.person_detector.track(img, threshold=self.confidence_threshold)
        
        # 2. Detect Faces
        faces = self.detect_faces(img)
        
        if self.debug_mode and len(faces) > 0:
            print(f"\n--- Faces Detected: {len(faces)} ---")
            for f in faces:
                print(f"   > Raw Face: {f['name']} (Score: {f['score']:.4f})")

        marshall_candidates = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            # Filter partial bodies
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15):
                continue
            
            marshall_score = self._calculate_marshall_score(bbox.tolist(), mask, img)
            marshall_candidates.append({
                'index': i,
                'track_id': track_id,
                'score': marshall_score,
                'bbox': bbox,
                'mask': mask
            })
        
        # NEW: Find the SINGLE best Marshall candidate (highest score)
        best_marshall = None
        if marshall_candidates:
            # Sort by score (highest first)
            marshall_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Take the highest scorer if they meet minimum threshold
            top_candidate = marshall_candidates[0]
            
            # Use temporal smoothing to confirm
            if self._determine_marshall_with_temporal_smoothing(
                top_candidate['track_id'], 
                top_candidate['score']
            ):
                best_marshall = top_candidate
                self.marshall_track_id = top_candidate['track_id']
                
                if self.debug_mode:
                    print(f"[Marshall Confirmed] ID {best_marshall['track_id']} (Score: {best_marshall['score']:.2f})")
        
        detected_athletes = []
        used_faces_indices = set()
        
        # 3. Process each Tracked Person
        for i in range(len(detections)):
            bbox = detections.xyxy[i].astype(int)
            mask = detections.mask[i] if detections.mask is not None else None
            conf = float(detections.confidence[i])
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            # Filter partial bodies
            if bbox[3] > (img_h * 0.95) and (bbox[3] - bbox[1]) < (img_h * 0.15):
                continue
            
            # NEW: Check if this is THE Marshall (using our single best candidate)
            is_marshall = (best_marshall is not None and best_marshall['index'] == i)
            
            marshall_score = next((c['score'] for c in marshall_candidates if c['index'] == i), 0.0)
            
            if is_marshall:
                # Clean registry if this track was previously an athlete
                if track_id in self.athlete_registry:
                    if self.debug_mode:
                        print(f"[Fix] ID {track_id} confirmed as Marshall. Cleaning athlete registry.")
                    del self.athlete_registry[track_id]
                
                # Create Marshall object and skip athlete processing
                marshall = DetectedAthlete(
                    name="MARSHALL",
                    track_id=track_id,
                    person_bbox=bbox.tolist(),
                    mask=mask,
                    center_x=(bbox[0] + bbox[2]) / 2,
                    confidence=conf,
                    face_bbox=None,
                    is_front_row=False,
                    debug_marshall_score=marshall_score
                )
                detected_athletes.append(marshall)
                continue

            # ATHLETE PROCESSING (unchanged from original)
            assigned_name = "Unknown"
            
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
                    if f is matched_face: 
                        used_faces_indices.add(idx)

            debug_log = f"[Track {track_id}] "
            
            if track_id != -1 and track_id in self.athlete_registry:
                current_registry_name = self.athlete_registry[track_id]
                debug_log += f"Mem: '{current_registry_name}'. "
                
                if new_face_name != "Unknown":
                    debug_log += f"Face found: '{new_face_name}' ({new_face_score:.2f}). "
                    if new_face_name == current_registry_name:
                        assigned_name = current_registry_name
                        debug_log += "MATCH -> Confirmed."
                    else:
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

            elif track_id != -1 and new_face_name != "Unknown":
                self.athlete_registry[track_id] = new_face_name
                assigned_name = new_face_name
                debug_log += f"New Track -> Assigned '{new_face_name}'."
            else:
                debug_log += "Unknown."

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
                debug_raw_face_name=new_face_name,
                debug_marshall_score=marshall_score
            )
            detected_athletes.append(athlete)
        
        # 4. Depth filtering
        if check_depth and detected_athletes:
            front, back = self.depth_analyzer.filter_front_row_athletes(
                detected_athletes, img_h, verbose=False
            )
            for a in front: 
                a.is_front_row = True
            for a in back: 
                a.is_front_row = False
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
            
            # NEW: Add Marshall score in debug mode
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