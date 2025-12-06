from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import os

class YOLOSegmentationModel:
    def __init__(self, model_path='yolo11l-seg.pt', tracker_config: str = None):
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.classes = self.model.names

        # Default to ReID-enabled tracker
        self.tracker_config = (
            tracker_config if tracker_config is not None else "custom_botsort.yaml"
        )

        # Fallback path
        self.default_tracker = "botsort.yaml"

        # Warn if custom config missing
        if not os.path.exists(self.tracker_config):
            print(
                f"[WARNING] Tracker config '{self.tracker_config}' not found! "
                f"Falling back to '{self.default_tracker}'."
            )
            self.active_tracker = self.default_tracker
        else:
            self.active_tracker = self.tracker_config

    def track(self, image, threshold: float = 0.5) -> sv.Detections:
        """
        Runs BoT-SORT with ReID enabled via custom config.
        Falls back when missing.
        """
        results = self.model.track(
            image,
            conf=threshold,
            persist=True,
            verbose=False,
            retina_masks=True,
            tracker=self.active_tracker
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # keep only class = person
        detections = detections[detections.class_id == 0]

        return detections
