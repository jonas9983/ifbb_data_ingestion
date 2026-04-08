import torch
from ultralytics import YOLO
import supervision as sv
import os

class YOLOSegmentationModel:
    def __init__(self, model_path='yolo11l-seg.pt', tracker_config: str = None):
        # Detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO loading on device: {self.device}")
        
        self.model = YOLO(model_path).to(self.device) # Explicitly move model to GPU
        self.classes = self.model.names
        
        self.tracker_config = tracker_config if tracker_config is not None else "custom_botsort.yaml"
        self.default_tracker = "botsort.yaml"
        self.active_tracker = self.tracker_config if os.path.exists(self.tracker_config) else self.default_tracker

    def track(self, image, threshold: float = 0.5) -> sv.Detections:
        # Pass the device to the track method
        results = self.model.track(
            image,
            conf=threshold,
            persist=True,
            verbose=False,
            retina_masks=True,
            tracker=self.active_tracker,
            device=self.device,
            half = true
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        return detections[detections.class_id == 0]
