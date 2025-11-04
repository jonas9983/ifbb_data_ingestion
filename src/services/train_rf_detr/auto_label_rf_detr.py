import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
import torch
import numpy as np

# --- Placeholder for the 13 specific user folders (Categories) ---
ALL_USER_FOLDERS = [
    "ahmd_ashkanani", 
    "angelcalderonfrias", 
    "fabriciomoreirapro", 
    "hideyamagishi", 
    "keone_prodigy", 
    "kerrith_bajjo", 
    "melnikovifbb", 
    "nassersayed_ifbbpro", 
    "olehkryvyi", 
    "shaunclarida", 
    "theradoslavangelov", 
    "venom_from_ukraine", 
    "vitorportopro"
]

class RF_Detr_AutoLabeler:
    """
    Performs object detection inference using RFDETRMedium .predict(), automatically
    labels detections based on the user folder name, and saves the output to Label Studio
    JSON format with predictions.
    """
    
    BASE_COCO_PERSON_CLASS_ID = 1
    DEFAULT_OUTPUT_BASE = "data/annotations/"

    def __init__(self, input_dir, confidence_threshold=0.5, image_url_prefix=""):
        self.input_path = Path(input_dir).resolve()
        self.confidence_threshold = confidence_threshold
        self.image_url_prefix = image_url_prefix
        self.model = None

        self.target_user = self.input_path.name
        self.output_path = Path(self.DEFAULT_OUTPUT_BASE) / self.target_user / "label_studio_predictions.json"
        
        self.categories_map = {name: i + 1 for i, name in enumerate(ALL_USER_FOLDERS)}
        self.TARGET_CATEGORY_ID = self.categories_map.get(self.target_user)
        
        if self.TARGET_CATEGORY_ID is None:
            raise ValueError(
                f"Target user folder '{self.target_user}' is not in the predefined list of users. "
                "Check ALL_USER_FOLDERS list."
            )
        
        print(f"Target User: **{self.target_user}** (Category: {self.target_user})")
        print(f"Output Path (Derived): **{self.output_path}**")

    def _initialize_model(self):
        """Initializes the RFDETRMedium model."""
        try:
            self.model = RFDETRMedium() 
            print("RFDETRMedium model initialized")
        except Exception as e:
            print(f"Error loading RFDETRMedium: {e}. Falling back to RFDETRBase.")
            try:
                from rfdetr import RFDETRBase
                self.model = RFDETRBase()
            except Exception as e_base:
                print(f"Failed to load RFDETRBase as fallback: {e_base}")
                self.model = None
                return

        try:
            self.model.optimize_for_inference() 
            print("Model successfully optimized for inference.")
        except Exception as e:
            print(f"Optimization failed: {e}")
            print("Inference will proceed using the un-optimized model.")

    def _get_image_files(self):
        """Collects all image files from the input directory."""
        if not self.input_path.is_dir():
            print(f"Error: Input directory not found at {self.input_path}")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in self.input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {self.input_path}.")
        return image_files

    def _generate_label_studio_json(self, all_detections, image_files):
        """Assembles all detections into Label Studio JSON format with predictions."""
        
        label_studio_tasks = []
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    original_width, original_height = img.size
            except Exception as e:
                print(f"Error reading image {img_file.name}: {e}. Skipping.")
                continue
            
            # Construct the image URL - either use prefix or local path
            if self.image_url_prefix:
                image_url = f"{self.image_url_prefix}/{img_file.name}"
            else:
                # Use relative path or file:// URL for local files
                image_url = f"/data/local-files/?d={self.target_user}/{img_file.name}"
            
            # Create the task structure
            task = {
                "data": {
                    "image": image_url
                },
                "predictions": []
            }
            
            # Check if we have detections for this image
            detections = all_detections.get(str(img_file))
            
            if detections and len(detections.xyxy) > 0:
                prediction_results = []
                
                for i in range(len(detections.xyxy)):
                    bbox_xyxy = detections.xyxy[i]
                    confidence = detections.confidence[i]
                    
                    # Handle both numpy arrays and torch tensors
                    if isinstance(bbox_xyxy, torch.Tensor):
                        bbox_xyxy = bbox_xyxy.cpu().numpy()
                    elif not isinstance(bbox_xyxy, np.ndarray):
                        bbox_xyxy = np.array(bbox_xyxy)
                    
                    if isinstance(confidence, torch.Tensor):
                        confidence = confidence.item()
                    elif isinstance(confidence, np.ndarray):
                        confidence = float(confidence)
                    else:
                        confidence = float(confidence)

                    x_min, y_min, x_max, y_max = bbox_xyxy
                    
                    # Convert to Label Studio format (percentage of image dimensions)
                    x_percent = (float(x_min) / original_width) * 100
                    y_percent = (float(y_min) / original_height) * 100
                    width_percent = ((float(x_max) - float(x_min)) / original_width) * 100
                    height_percent = ((float(y_max) - float(y_min)) / original_height) * 100
                    
                    # Create Label Studio rectangle annotation
                    result_item = {
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                            "rotation": 0,
                            "rectanglelabels": [self.target_user]
                        },
                        "id": f"bbox_{i}",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels"
                    }
                    
                    prediction_results.append(result_item)
                
                # Add predictions to task
                if prediction_results:
                    task["predictions"].append({
                        "result": prediction_results,
                        "score": float(np.mean([float(detections.confidence[i]) 
                                               if isinstance(detections.confidence[i], (np.ndarray, torch.Tensor))
                                               else detections.confidence[i]
                                               for i in range(len(detections.confidence))]))
                    })
            
            label_studio_tasks.append(task)
        
        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(label_studio_tasks, f, indent=2)
        
        total_predictions = sum(len(task.get("predictions", [])) for task in label_studio_tasks)
        total_boxes = sum(len(pred.get("result", [])) 
                         for task in label_studio_tasks 
                         for pred in task.get("predictions", []))
        
        print(f"\nSuccessfully generated Label Studio JSON file:")
        print(f"  - {len(label_studio_tasks)} tasks")
        print(f"  - {total_boxes} bounding box predictions")
        print(f"  - Category: **{self.target_user}**")
        print(f"  - Output: {self.output_path}")

    def run(self):
        """Executes the auto-labeling process."""
        if not self.TARGET_CATEGORY_ID:
            return

        self._initialize_model()
        if not self.model:
            return

        image_files = self._get_image_files()
        if not image_files:
            return

        print(f"Processing {len(image_files)} images...")
        
        all_detections = {}

        for img_file in tqdm(image_files, desc="Running Inference"):
            try:
                image = Image.open(img_file).convert("RGB")
                detections = self.model.predict(image, threshold=self.confidence_threshold)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}. Skipping.")
                continue

            # Handle class_id as either tensor or numpy array
            class_id = detections.class_id
            if isinstance(class_id, torch.Tensor):
                person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
            elif isinstance(class_id, np.ndarray):
                person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
            else:
                person_mask = [cid == self.BASE_COCO_PERSON_CLASS_ID for cid in class_id]
            
            person_detections = detections[person_mask]
            
            if len(person_detections) > 0:
                all_detections[str(img_file)] = person_detections

        # Generate Label Studio JSON (even if no detections, so empty tasks can be labeled)
        self._generate_label_studio_json(all_detections, image_files)
        
        if not all_detections:
            print(f"\nNote: No 'person' detections found above {self.confidence_threshold} confidence.")
            print(f"Empty tasks have been created for manual labeling in Label Studio.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Auto-label images using RFDETRMedium().predict() and save results in Label Studio JSON format.'
    )
    parser.add_argument(
        '--input', 
        required=True,
        help='Directory containing the input images (MUST be the user folder, e.g., data/temp/keone_prodigy/).'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.5, 
        help='Confidence threshold for saving detections (default: 0.5).'
    )
    parser.add_argument(
        '--url-prefix',
        type=str,
        default="",
        help='URL prefix for images (e.g., http://localhost:8080/data). Leave empty for local file paths.'
    )
    
    args = parser.parse_args()
    
    try:
        labeler = RF_Detr_AutoLabeler(
            input_dir=args.input,
            confidence_threshold=args.conf,
            image_url_prefix=args.url_prefix
        )
        labeler.run()
    except ValueError as e:
        print(f"Initialization Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")