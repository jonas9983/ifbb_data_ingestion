from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import cv2

class YOLOSegmentationModel:
    def __init__(self, model_size='m'):
        """
        Initializes YOLOv8 Segmentation.
        model_size options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        'm' (medium) is a great balance for athlete detection.
        """
        print(f"Loading YOLOv8{model_size}-seg model...")
        # This will auto-download 'yolov8m-seg.pt' to your current folder
        self.model = YOLO(f"yolov8{model_size}-seg.pt")
        self.classes = self.model.names

    def predict(self, image, threshold: float = 0.5) -> sv.Detections:
        """
        Runs inference and returns supervision.Detections with masks.
        """
        # YOLO accepts PIL images, numpy arrays, or file paths
        results = self.model(
            image, 
            conf=threshold, 
            verbose=False,
            retina_masks=True # High quality masks
        )[0]

        # Convert to Supervision Detections format
        detections = sv.Detections.from_ultralytics(results)

        # Filter for Class ID 0 (Person) only
        # In COCO dataset, 'person' is always class_id 0
        detections = detections[detections.class_id == 0]

        return detections