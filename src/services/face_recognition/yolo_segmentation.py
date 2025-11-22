from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import cv2

class YOLOSegmentationModel:
    def __init__(self, model_path='yolov11l-seg.pt'): 
        """
        Initializes YOLO Segmentation (Works for v8 and v11).
        Recommended: Use 'yolo11l-seg.pt' for best performance.
        """
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def track(self, image, threshold: float = 0.5) -> sv.Detections:
        """
        Runs TRACKING instead of just prediction.
        'persist=True' is critical: it keeps ID #1 as ID #1 in the next frame.
        """
        # Run inference with tracking
        results = self.model.track(
            image, 
            conf=threshold, 
            persist=True,
            verbose=False,
            retina_masks=True,
            tracker="bytetrack.yaml" # Standard, robust tracker
        )[0]

        # Convert to Supervision Detections
        # sv.Detections.from_ultralytics automatically extracts 'tracker_id'
        detections = sv.Detections.from_ultralytics(results)

        # Filter for Class ID 0 (Person) only
        detections = detections[detections.class_id == 0]

        return detections