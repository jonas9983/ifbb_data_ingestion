import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rfdetr import RFDETRMedium
import torch
import numpy as np

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
    BASE_COCO_PERSON_CLASS_ID = 1

    def __init__(self, model, input_dir, split_name="test", target_user_name: str = None, 
                 confidence_threshold=0.5, image_url_prefix="", 
                 use_custom_classes=False, class_names=None):
        """
        Args:
            model: The RFDETR model
            input_dir: Directory containing images
            split_name: Name of the split (train/val/test)
            target_user_name: Target label name (only used if use_custom_classes=False)
            confidence_threshold: Confidence threshold for detections
            image_url_prefix: Optional URL prefix for Label Studio
            use_custom_classes: If True, use class_id to determine labels from class_names
            class_names: List of class names (e.g., ALL_USER_FOLDERS for custom model)
        """
        self.input_path = Path(input_dir).resolve()
        self.confidence_threshold = confidence_threshold
        self.image_url_prefix = image_url_prefix
        self.model = model
        self.split_name = split_name
        self.use_custom_classes = use_custom_classes
        self.class_names = class_names if class_names else ALL_USER_FOLDERS

        # For backward compatibility with single-label mode
        self.target_user = target_user_name if target_user_name is not None else split_name
        
        self.categories_map = {name: i + 1 for i, name in enumerate(ALL_USER_FOLDERS)}
        
        if self.target_user in self.categories_map:
            self.TARGET_CATEGORY_ID = self.categories_map[self.target_user]
            self.is_athlete_folder = True
        else:
            self.TARGET_CATEGORY_ID = None
            self.is_athlete_folder = False
            
        if use_custom_classes:
            print(f"Info: Using custom classes mode. Will detect {len(self.class_names)} classes.")
        else:
            print(f"Info: Using single-label mode. Labeling detected objects as '{self.target_user}'.")

    def _get_image_files(self):
        if not self.input_path.is_dir():
            print(f"Error: Input directory not found at {self.input_path}")
            return []

        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in self.input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {self.input_path}.")
        return image_files

    def _format_task(self, img_file, detections):
        try:
            with Image.open(img_file) as img:
                original_width, original_height = img.size
        except Exception as e:
            print(f"Error reading image {img_file.name}: {e}. Skipping.")
            return None

        if self.is_athlete_folder:
            relative_storage_path = f"{self.split_name}/{self.target_user}/{img_file.name}"
        else:
            relative_storage_path = f"{self.split_name}/{img_file.name}"

        local_file_url_part = f"/data/local-files/?d={relative_storage_path}"

        if self.image_url_prefix:
            base_url = self.image_url_prefix.rstrip('/')
            image_url = f"{base_url}{local_file_url_part}"
        else:
            image_url = local_file_url_part

        task = {
            "data": {"image": image_url},
            "predictions": []
        }

        if detections and len(detections.xyxy) > 0:
            prediction_results = []

            for i in range(len(detections.xyxy)):
                bbox_xyxy = detections.xyxy[i]
                confidence = detections.confidence[i]
                
                # Get the class_id for this detection
                class_id = detections.class_id[i]

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
                    
                if isinstance(class_id, torch.Tensor):
                    class_id = class_id.item()
                elif isinstance(class_id, np.ndarray):
                    class_id = int(class_id)
                else:
                    class_id = int(class_id)

                x_min, y_min, x_max, y_max = bbox_xyxy

                x_percent = (float(x_min) / original_width) * 100
                y_percent = (float(y_min) / original_height) * 100
                width_percent = ((float(x_max) - float(x_min)) / original_width) * 100
                height_percent = ((float(y_max) - float(y_min)) / original_height) * 100

                # Determine the label based on mode
                if self.use_custom_classes:
                    # Use the class_id to get the actual class name
                    label = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                else:
                    # Use the single target_user label (backward compatible)
                    label = self.target_user

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
                        "rectanglelabels": [label]
                    },
                    "id": f"bbox_{i}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels"
                }

                prediction_results.append(result_item)

            if prediction_results:
                task["predictions"].append({
                    "result": prediction_results,
                    "score": float(np.mean([
                        float(detections.confidence[i])
                        if isinstance(detections.confidence[i], (np.ndarray, torch.Tensor))
                        else detections.confidence[i]
                        for i in range(len(detections.confidence))
                    ]))
                })

        return task

    def run(self, return_detections=False):
        """
        Run inference on all images.
        
        Args:
            return_detections: If True, return both tasks and raw detections for visualization
        
        Returns:
            all_tasks: List of Label Studio formatted tasks
            all_detections: (Optional) List of (image_path, detections) tuples for visualization
        """
        image_files = self._get_image_files()
        if not image_files:
            return ([], []) if return_detections else []

        all_tasks = []
        all_detections = []  # Store raw detections for visualization

        for img_file in tqdm(image_files, desc=f"Inference: {self.split_name}"):
            try:
                image = Image.open(img_file).convert("RGB")
                detections = self.model.predict(image, threshold=self.confidence_threshold)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}. Skipping.")
                continue

            # Store raw detections BEFORE filtering (for visualization)
            if return_detections:
                all_detections.append((img_file, detections))

            # Filter detections if needed
            filtered_detections = detections
            if not self.use_custom_classes:
                # Legacy mode: filter for COCO person class only
                class_id = detections.class_id
                if isinstance(class_id, torch.Tensor):
                    person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
                elif isinstance(class_id, np.ndarray):
                    person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
                else:
                    person_mask = [cid == self.BASE_COCO_PERSON_CLASS_ID for cid in class_id]
                
                filtered_detections = detections[person_mask]

            task = self._format_task(img_file, filtered_detections)
            if task:
                all_tasks.append(task)

        if return_detections:
            return all_tasks, all_detections
        return all_tasks


def process_entire_dataset(base_dir, output_dir, confidence_threshold=0.5, image_url_prefix=""):
    """Original function for COCO person detection (backward compatible)"""
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    splits = ["train", "val", "test"]

    print(f"Running auto-labeling for dataset in {base_dir}")

    model = RFDETRMedium()
    model.optimize_for_inference()

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_path = base_dir / split
        if not split_path.exists():
            print(f"Split not found: {split_path}")
            continue

        combined_tasks = []
        directories_to_process = []
        
        subdirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if subdirs:
            for athlete_dir in subdirs:
                if athlete_dir.name in ALL_USER_FOLDERS:
                    directories_to_process.append((athlete_dir, athlete_dir.name, athlete_dir.name))
        else:
            target_name = split_path.name
            directories_to_process.append((split_path, target_name, f"root ({target_name})"))

        if not directories_to_process:
            print(f"Skipping {split}: No relevant images or subfolders found.")
            continue

        for input_dir, target_user_name, descriptive_name in directories_to_process:
            print(f"Processing {split}/{descriptive_name}")
            
            labeler = RF_Detr_AutoLabeler(
                model=model,
                input_dir=input_dir,
                split_name=split,
                target_user_name=target_user_name,
                confidence_threshold=confidence_threshold,
                image_url_prefix=image_url_prefix
            )
            tasks = labeler.run()
            combined_tasks.extend(tasks)
        
        output_path = output_dir / f"{split}.json"
        with open(output_path, 'w') as f:
            json.dump(combined_tasks, f, indent=2)

        print(f"Saved {len(combined_tasks)} tasks to {output_path}")


def visualize_detections(image_path, detections, class_names, max_size=800):
    """
    Visualize detections on an image using supervision library.
    
    Args:
        image_path: Path to the image file
        detections: Raw detections from model.predict()
        class_names: List of class names (e.g., ALL_USER_FOLDERS)
        max_size: Maximum dimension for thumbnail (default 800)
    
    Returns:
        annotated_image: PIL Image with bounding boxes and labels
    """
    try:
        import supervision as sv
    except ImportError:
        print("Please install supervision: pip install supervision")
        return None
    
    
    image = Image.open(image_path)
    
    # Setup colors and styles
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True
    )
    
    # Create labels with class names and confidence
    labels = [
        f"{class_names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    
    # Annotate image
    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    
    # Resize for display
    annotated_image.thumbnail((max_size, max_size))
    
    return annotated_image


def visualize_batch(detections_list, class_names, output_dir=None, max_images=5):
    """
    Visualize multiple images with detections.
    
    Args:
        detections_list: List of (image_path, detections) tuples from labeler.run(return_detections=True)
        class_names: List of class names
        output_dir: Optional directory to save visualizations
        max_images: Maximum number of images to visualize
    
    Returns:
        List of annotated PIL Images
    """
    annotated_images = []
    
    for i, (img_path, detections) in enumerate(detections_list[:max_images]):
        print(f"Visualizing {img_path.name}...")
        
        annotated = visualize_detections(img_path, detections, class_names)
        if annotated:
            annotated_images.append(annotated)
            
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                save_path = output_path / f"annotated_{img_path.name}"
                annotated.save(save_path)
                print(f"  Saved to {save_path}")
    
    return annotated_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Auto-label train/val/test images using RFDETR and save combined results per split.'
    )
    parser.add_argument('--input', required=True, help='Root directory containing train/val/test folders.')
    parser.add_argument('--output', default='data/annotations', help='Directory to save combined JSONs.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default=0.5)')
    parser.add_argument('--url-prefix', type=str, default="", help='Optional URL prefix for Label Studio.')

    args = parser.parse_args()

    process_entire_dataset(
        base_dir=args.input,
        output_dir=args.output,
        confidence_threshold=args.conf,
        image_url_prefix=args.url_prefix
    )