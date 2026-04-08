import cv2
import json
import numpy as np
from typing import Tuple, Optional
from src.services.faces.face_recognizer import FaceRecognizer
from src.services.athlete_tracking.detection import AthleteDetector
from src.services.athlete_tracking.swap_detection import TrackingEventLogger
from src.services.athlete_tracking.yolo_segmentation import YOLOSegmentationModel
from src.services.helpers.frame_logger import FrameLogger

class AthletePositionTracker:
    def __init__(self, db_path: str, threshold: float = 0.35, confidence_threshold: float = 0.5, tracker_config: str = None, debug_mode: bool = False):
        print("Loading FaceRecognizer...")
        face_recognizer = FaceRecognizer(db_path, threshold=threshold)
        print("Recognizer loaded.")
        
        person_model = YOLOSegmentationModel(model_path='yolo11l-seg.pt', tracker_config=tracker_config)
        print("Person segmentation model loaded.")
        
        self.detector = AthleteDetector(face_recognizer, person_model, confidence_threshold=confidence_threshold, debug_mode=debug_mode)
        self.swap_detector = SwapDetector()
        self.frame_logger = FrameLogger()
        self.last_gray_frame: Optional[np.ndarray] = None
        self.cut_threshold: float = 25.0

    def _check_for_camera_cut(self, img: np.ndarray, frame_name: str) -> Tuple[bool, float]:
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.last_gray_frame is None:
            self.last_gray_frame = current_gray
            return False, 0.0

        diff = cv2.absdiff(current_gray, self.last_gray_frame)
        avg_diff = float(np.mean(diff))
        is_cut = avg_diff > self.cut_threshold
        self.last_gray_frame = current_gray
        
        if is_cut:
            print(f"✂️ CAMERA CUT: {frame_name} (diff={avg_diff:.2f})")
            if hasattr(self.detector.person_detector.model, 'predictor') and hasattr(self.detector.person_detector.model.predictor, 'trackers'):
                for tracker in self.detector.person_detector.model.predictor.trackers:
                    tracker.reset()
            self.detector.athlete_registry.clear()
            self.detector.marshall_track_id = -1
            
        return is_cut, avg_diff

    def process_frame(self, img, frame_name: str, frame_number: int, show_person_bbox: bool = True, filter_front_row: bool = True):
        is_cut, avg_diff = self._check_for_camera_cut(img, frame_name)
        if is_cut:
            self.swap_detector.reset_state(frame_number, frame_name, "camera_cut")
            
        all_athletes = self.detector.detect_and_associate(img, check_depth=filter_front_row)
        
        valid_front_row_athletes = [a for a in all_athletes if a.is_front_row and a.name not in ["Unknown", "MARSHALL"]]
        current_positions = {athlete.name: athlete.center_x for athlete in valid_front_row_athletes}
        
        swap_info = self.swap_detector.update_state(current_positions, frame_name, frame_number)
        swap_info['positions'] = current_positions
        
        self.frame_logger.log_frame(
            frame_number=frame_number, frame_name=frame_name, athletes=all_athletes,
            raw_faces=[], is_camera_cut=is_cut, avg_pixel_diff=avg_diff,
            registry_state=self.detector.athlete_registry.copy(),
            marshall_track_id=self.detector.marshall_track_id, swap_info=swap_info, frame_dims=img.shape[:2]
        )
        
        self.detector.draw_annotations(img, all_athletes, show_person_bbox)
        return all_athletes
    
    def export_events(self, output_path: str):
        stats = self.swap_detector.get_statistics()
        events_data = {'summary': stats, 'events': [event.to_dict() for event in self.swap_detector.events]}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        print(f"\n Events exported to: {output_path}")
    
    def export_comprehensive_data(self, output_path: str):
        self.frame_logger.export_to_json(output_path)
    
    def print_summary(self):
        self.swap_detector.print_summary()